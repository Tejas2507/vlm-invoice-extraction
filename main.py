import random
import os
import sys
import json
import time
import re
import argparse
import ast 
import gc
import numpy as np
import torch
from rapidfuzz import process, fuzz
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from ultralytics import YOLO
from unsloth import FastVisionModel

# ==========================================
# CONFIGURATION
# ==========================================
# PATHS UPDATED FOR SUBMISSION
UNSLOTH_MODEL_ID = "unsloth/Qwen3-VL-8B-Instruct-bnb-4bit" # Downloads from HF Hub
VISION_MODEL_PATH = os.path.join(os.path.dirname(__file__), "utils", "best.pt")

# OPTIMIZATION: SEPARATE RESOLUTIONS
YOLO_IMG_SIZE = 1024  # High res for detection accuracy
MAX_VLM_SIZE = 640    # Low res for VLM speed (saves tokens)

# ENABLE BOOTSTRAPPING (Double Check)
ENABLE_BOOTSTRAP = True 

BASE_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "sample_output")
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    
KNOWN_BRANDS = [
    "mahindra", "swaraj", "sonalika", "massey ferguson", "mf", "tafe", 
    "escorts", "john deere", "eicher", "new holland", "kubota", "farmtrac", 
    "powertrac", "captain tractors", "force motors", "preet tractors", 
    "indo farm", "same deutz fahr", "ace", "vst shakti", "solis", "hav", 
    "autonxt", "cellestial", "trakstar", "maxgreen", "marut", "sukoon", 
    "montra", "hindustan", "kartar", "field marshall", "ford", "hmt", 
    "mahindra gujarat", "vst", "force", "captain"
]

# ==========================================
# 1. SETUP & LOADERS
# ==========================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

vision_model = None
model = None
tokenizer = None

def load_models():
    global vision_model, model, tokenizer
    # Silent loading if possible, or minimal stderr
    try:
        vision_model = YOLO(VISION_MODEL_PATH)
    except:
        sys.exit(1)

    try:
        model, tokenizer = FastVisionModel.from_pretrained(
            UNSLOTH_MODEL_ID,
            load_in_4bit=True,
            device_map="auto",
        )
        FastVisionModel.for_inference(model)
    except Exception as e:
        sys.stderr.write(f"Model Load Error: {e}\n")
        sys.exit(1)


# ==========================================
# 2.5. WARMUP ROUTINE
# ==========================================
def warmup_pipeline():
    """
    Runs a dummy inference to 'burn in' CUDA kernels and allocate buffers.
    This prevents the 'Double Time/RAM' spike on the first real image.
    """
    try:
        # Dummy Black Image
        vision_model.predict(Image.new('RGB', (64, 64), color='black'), verbose=False)
        
        # 2. Warmup Qwen (Minimal)
        dummy_img = Image.new('RGB', (64, 64), color='black')
        messages = [{"role": "user", "content": [{"type": "image", "image": dummy_img}, {"type": "text", "text": "test"}]}]
        text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        inputs = tokenizer(images=[dummy_img], text=[text], padding=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=1)
    except Exception as e:
        pass

def validate_extraction(data):
    """
    Programmatic Guardrails. Returns (is_valid, reasons).
    """
    errors = []
    
    # 1. HP Check
    try:
        hp = float(str(data.get("horse_power", 0)).replace('HP','').strip())
        if hp < 10 or hp > 200:
            errors.append(f"HP {hp} is out of realistic range (10-200).")
    except:
        pass # If not numeric, VLM might have output text, which Pass 2 catches.

    # 2. Cost Check
    try:
        cost = float(str(data.get("asset_cost", 0)).replace(',','').strip())
        if cost < 20000 or cost > 2000000:
             if cost > 0: errors.append(f"Asset Cost {cost} seems invalid (Strict Range: 20k - 20L).")
        elif cost < 100000:
             # User Request: Flag < 1L for verification by Pass 2, but don't reject outright if Pass 2 confirms.
             errors.append(f"FLAG: Asset Cost {cost} is LOW (< 1L). Verify.")
    except:
        pass

    # 3. Model Name "Brand Pollution" Check - AUTO FIX
    m_name = str(data.get("model_name", ""))
    m_name_lower = m_name.lower()
    for brand in KNOWN_BRANDS:
        if brand in m_name_lower:
            # User Request: Strip programmatically without calling Supervisor
            # Case-insensitive replace is tricky, so simplified approach:
            pattern = re.compile(re.escape(brand), re.IGNORECASE)
            clean_name = pattern.sub("", m_name).strip()
            # Update data IN-PLACE
            data["model_name"] = clean_name
            # No error appended, so we don't trigger Pass 2 just for this!
            break

    # 4. HP Pollution Check - AUTO FIX
    # Strip "50 HP", "47 H.P" (2 digits only) from model name if present
    # Prevents removing "575" (Model) while removing "47 HP" (Power)
    m_name = str(data.get("model_name", ""))
    
    # 4a. Strip Special Characters (!@#$%^&*())
    m_name = re.sub(r'[!@#$%^&*()]', '', m_name).strip()
    
    # 4b. Strip HP Pattern (Suffix "42 HP" OR Prefix "HP 42")
    hp_pattern = re.compile(r'(\b\d{2}\s*H\.?P\.?\b)|(\bH\.?P\.?\s*\d{2}\b)', re.IGNORECASE)
    if hp_pattern.search(m_name):
        clean_name = hp_pattern.sub("", m_name).strip()
        data["model_name"] = clean_name
    else:
        # If no HP stripping happened but we stripped special chars, update it
        data["model_name"] = m_name

    return (len(errors) == 0), errors

def final_clean_model_name(model_name):
    """
    Safety net regex to strip brand names if LLM fails 2 times.
    Preserves original case, removes brands/HP/Special Chars.
    """
    if not model_name: return ""
    
    # Start with original model_name (NO .lower())
    clean_name = model_name
    
    # 1. Strip Brands (Case Insensitive)
    for brand in KNOWN_BRANDS:
        pattern = re.compile(re.escape(brand), re.IGNORECASE)
        clean_name = pattern.sub("", clean_name).strip()
        
    # 2. Strip Special Chars
    clean_name = re.sub(r'[!@#$%^&*()]', '', clean_name).strip()
        
    # 3. Strip HP patterns (Strictly 2 digits, Prefix or Suffix)
    hp_pattern = re.compile(r'(\b\d{2}\s*H\.?P\.?\b)|(\bH\.?P\.?\s*\d{2}\b)', re.IGNORECASE)
    clean_name = hp_pattern.sub("", clean_name).strip()
    
    return clean_name

def detect_objects_yolo(image_path):
    try:
        results = vision_model.predict(image_path, conf=0.10, iou=0.45, imgsz=YOLO_IMG_SIZE, verbose=False)
        detections = {"header": None, "detail": None, "stamp": None, "signature": None}
        
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = vision_model.names[cls_id].lower()
                
                key = None
                if "header" in cls_name: key = "header"
                elif "detail" in cls_name: key = "detail"
                elif "stamp" in cls_name: key = "stamp"
                elif "signature" in cls_name: key = "signature"
                
                if key:
                    conf = float(box.conf[0])
                    coords = [int(x) for x in box.xyxy[0].tolist()]
                    if detections[key] is None or conf > 0.5: 
                        detections[key] = coords 
        return detections
    except:
        return {}

# ==========================================
# 3. UNSLOTH INFERENCE
# ==========================================
def run_qwen_inference(pil_images, prompt_text):
    """
    Accepts LIST of PIL IMAGES.
    """
    torch.cuda.empty_cache()
    gc.collect()
    
    try:
        messages = [{
            "role": "user",
            "content": [{"type": "image", "image": img} for img in pil_images] + 
                       [{"type": "text", "text": prompt_text}]
        }]

        # Resize for VLM speed if needed
        final_images = []
        for img in pil_images:
            if max(img.size) > MAX_VLM_SIZE:
                ratio = MAX_VLM_SIZE / max(img.size)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            final_images.append(img)

        text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        inputs = tokenizer(images=final_images, text=[text], padding=True, return_tensors="pt").to("cuda")
        
        outputs = model.generate(
            **inputs, 
            max_new_tokens=100, 
            temperature=0.01, 
            do_sample=False,
            use_cache=True
        )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)
        ]
        out_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # --- ROBUST PARSING ---
        json_data = {}
        json_str = out_text.strip()
        
        if "```json" in json_str: json_str = json_str.split("```json")[1].split("```")[0]
        elif "```" in json_str: json_str = json_str.split("```")[1].split("```")[0]
        
        start = json_str.find('{')
        if start != -1:
            json_str = json_str[start:]
            end = json_str.rfind('}')
            if end == -1:
                json_str += '"}' 
                try: json.loads(json_str)
                except: json_str = json_str[:-2] + "}" 
            else:
                json_str = json_str[:end+1]

            try:
                json_data = json.loads(json_str)
            except:
                try: json_data = ast.literal_eval(json_str)
                except: json_data = {}
        
        return json_data, out_text
        
    except Exception as e:
        return {}, ""
        
# ==========================================
# 4. MAIN PIPELINE (THE GATEKEEPER)
# ==========================================
def process_invoice(image_path):
    start_time = time.perf_counter()
    filename = os.path.basename(image_path)
    root = os.path.splitext(filename)[0]
    
    # 1. YOLO
    detections = detect_objects_yolo(image_path)
    
    # 2. Image Prep
    pil_img = Image.open(image_path).convert("RGB")
    
    # --- VISUAL ANCHORING: Draw Header Box on Full Image ---
    anchored_full_img = pil_img.copy()
    header_bbox = detections.get("header")
    
    if header_bbox:
        draw = ImageDraw.Draw(anchored_full_img)
        # RED BOX for "Look Here"
        draw.rectangle(header_bbox, outline="red", width=5)
    
    # VLM Input List
    vlm_images = [anchored_full_img]

    # Detail Crop Preparation
    key = "detail"
    has_detail = False
    
    if detections[key]:
        x1, y1, x2, y2 = detections[key]
        if x2 > x1 and y2 > y1: # Safety check
            detail_crop = pil_img.crop((max(0, x1-10), max(0, y1-10), min(pil_img.width, x2+10), min(pil_img.height, y2+10)))
            # Optimization 4: Grayscale to reduce noise/tokens
            vlm_images.append(detail_crop.convert("L"))
            has_detail = True
    
    brand_string = ", ".join([b.capitalize() for b in KNOWN_BRANDS])
    
    # Pre-compute Dynamic Prompt Strings based on available images
    if has_detail:
        p_img2_desc = "- IMAGE 2 (MODEL ZOOM): LOOK HERE for the MODEL NAME, and HP."
        p_rule2_loc = "IMAGE 2 (Model Zoom)"
        p_rule3_loc = "IMAGE 2"
        p_img_list = "1. Full Invoice (With RED BOX hint)\n2. Model/Detail Zoom"
        p_verify_img2 = "- CHECK IMAGE 2 (Model Zoom) for the Model Name/HP."
    else:
        p_img2_desc = ""
        p_rule2_loc = "IMAGE 1 (Full Invoice)"
        p_rule3_loc = "IMAGE 1"
        p_img_list = "1. Full Invoice (With RED BOX hint FOR HEADER) look for MODEL NAME AND HP IN THE FULL IMAGE"
        p_verify_img2 = ""


    # =========================================================================
    # PASS 1: JUNIOR ANALYST (Structured Anchoring)
    # =========================================================================
    SYSTEM_PROMPT_1 = f"""You are an experienced Invoice Analyst. Extract fields into JSON.

[STRUCTURED ANCHORING]
You have {len(vlm_images)} images. Use them as follows:
- IMAGE 1 (FULL INVOICE): I have drawn a APPROX RED BOX around the Header. LOOK INSIDE THE RED BOX for the DEALER NAME AND STRICTLY OUTPUT VERNACULAR DEALER NAME IF PRESENT DONOT TRANSLITERATE.
 Also observe the OVERALL STRUCTURE LOOK FOR POTENTIAL MODEL NAME AND HP.
{p_img2_desc}

RULES:
1. dealer_name: The Business Name at the top. [use the RED BOX in IMAGE 1.] 
   - If the header is in Hindi/Vernacular (e.g. 'किसान ट्रैक्टर्स' , 'ಲಕ್ಷ್ಮಿ ಟ್ರೇಡರ್ಸ್' , અમન ટ્રેક્ટર્સ) Give the exact name as output.
   - If you see a lot of text in languages other than ENGLISH, Search the Header name in the other language.
   - Make sure you do not confuse Dealer name with company names.

2. model_name: Exact Model.
    - Check for suffixes like 'DI', 'RX', 'PLUS', 'XP', 'SUPER', 'PRO' , 'TECH' , 'MAX'.
    - Look for MODEL NAMES near company names like: {brand_string}.
    - EXAMPLE: IF 'Mahindra 575 DI' -> OUTPUT '575 DI'. IF 'Swaraj 744 FE' -> OUTPUT '744 FE'.
    - STRICTLY REMOVE THE BRAND NAME.
    - If ticked in a list, select the ticked row.
    - Model Name Should be in ENGLISH. If you see in any other language TRANSLITERATE. 

3. horse_power: Numeric HP.[STRICT RANGE - (10-200)] 
      - LOOK for fields like e.g "HP : 48" , "55 HP" , "HP = 39"
      - DO NOT infer from MODEL NAME

4. asset_cost: Total Amount (Numeric) [STRICT RANGE - [20,000 - 20,00,000] [If you see many USUALLY consider the highest one].

IMPORTANT - ANALYST NOTES:
- Briefly mention where you found the ASSET COST , Model Name and HP. (e.g., "Found Model 575 DI in Header", "HP inferred is explicitly hand written" ,).
- IF YOU ARE UNSURE about any field, START THE NOTE WITH "FLAG:" followed by the reason.
- IF text is blurry or ambiguous, START THE NOTE WITH "FLAG:".
- DO NOT to INFER HP from MODEL NAME. IF you are INFERRING , then 'FLAG' : HP is ambiguous
- IF ASSET COST IS lower than 100000 STRICTLY FLAG : 'ASSET COST IS LOW , VERIFY'.

OUTPUT FORMAT:
{{
  "dealer_name": "...", [STRICTLY DONOT TRANSLITERATE TO ENGLISH]
  "model_name": "...",  [STRICTLY DONOT PUT company_names/H.P AT THE FRONT]
  "horse_power": "...", [STRICT RANGE - (10 - 150) ]
  "asset_cost": "...", [STRICT RANGE - [50,000 - 20,00,000]
  "analyst_notes": "Found Model in Header. FLAG: HP is ambiguous..." [BE CRISP AND CLEAR DONOT EXPLAIN]
}}"""

    PROMPT_TEXT_1 = f"""{SYSTEM_PROMPT_1}

You are provided with {len(vlm_images)} images: 
{p_img_list}

Analyze them using the Anchoring rules above.
JSON OUTPUT:"""
    
    # We pass PIL images directly now, not paths
    data_pass1, raw_text_1 = run_qwen_inference(vlm_images, PROMPT_TEXT_1)
    
    # =========================================================================
    # THE GATEKEEPER 
    # =========================================================================
    notes = str(data_pass1.get("analyst_notes", "")).lower().strip()
    
    # --- PROGRAMMATIC VALIDATION ---
    is_valid_data, validation_errors = validate_extraction(data_pass1)
    
    # Validation Failures = AUTO FAIL
    if not is_valid_data:
         is_confident = False
         junior_notes = notes + " | SYSTEM ALERTS: " + "; ".join(validation_errors)
    else:
        # Keyword Scan for Uncertainty
        danger_words = ["flag:", "unsure", "unclear", "guess", "ambiguous", "illegible", "blur"]
        has_danger = any(w in notes for w in danger_words)
        
        # Missing Critical Fields check
        missing_fields = []
        if not data_pass1.get("dealer_name"): missing_fields.append("dealer_name")
        if not data_pass1.get("model_name"): missing_fields.append("model_name")
        
        if has_danger:
            is_confident = False
            junior_notes = notes
        elif missing_fields:
            is_confident = False
            junior_notes = f"Missing fields: {missing_fields}. {notes}"
        else:
            is_confident = True
            junior_notes = notes
    
    if not is_confident and data_pass1:
        # =====================================================================
        # PASS 2: SENIOR SUPERVISOR
        # =====================================================================
        # Send same images
        locked_dealer = data_pass1.get("dealer_name")

        # Prepare summary of Pass 1 for Supervisor to respect
        p1_summary = json.dumps({k:v for k,v in data_pass1.items() if k in ['dealer_name', 'model_name', 'horse_power', 'asset_cost']}, ensure_ascii=False)

        PROMPT_TEXT_2 = f"""You are a Senior Supervisor. Your job is to FIX ERRORS flagged by the system, but PRESERVE what is already correct.

[JUNIOR ANALYST FINDINGS]
The first analyst found this:
{p1_summary}

[DETECTED ISSUES]
The Analyst detected these specific findings above:
"{junior_notes}"

[YOUR ORDERS]
1. FIX THE FLAGGED ISSUES IMMEDIATELY:
   - If "HP out of range", FIND THE REAL HP. **LOOK AROUND THE MODEL NAME** in the Detail/Model Zoom image! It is often written nearby (e.g. "47 HP").
2. VERIFY DATA: Double check spelling and digits against the images.
   - CHECK IMAGE 1 (Red Box) for Dealer accuracy.
   {p_verify_img2}
5. FINAL OUTPUT: Must be valid JSON.

OUTPUT FORMAT:
JSON: {{
  "audit_check": "FIXED",
  "dealer_name": "Correct Dealer Name", [STRICTLY DONOT TRANSLITERATE TO ENGLISH]
  "model_name": "Correct Model Name", [LOOK AROUND THE MODEL NAME IN THE DETAIL/MODEL ZOOM IMAGE FOR MORE CLUE]
  "horse_power": [Numeric] [MUST BE INTEGER 10-150. NEVER > 150]
  "asset_cost": [Numeric] [STRICT RANGE - [50,000 - 20,00,000]] LOOK FOR things like "total cost" , 'Net Amount' etc.  
  "confidence_score": [Numeric] [MUST BE BETWEEN 0.90 - 1.00]
}}"""
        
        data_final, raw_text_2 = run_qwen_inference(vlm_images, PROMPT_TEXT_2)
        
        if not data_final.get("dealer_name"): 
             data_final["dealer_name"] = locked_dealer
            
    else:
        data_final = data_pass1

    # 5. Finalize
    f_model = final_clean_model_name(str(data_final.get("model_name")))
    
    # FINAL SAFETY NET FOR HP
    try: 
        f_hp = int(data_final.get("horse_power"))
        if f_hp > 100:
            s_hp = str(f_hp)
            if len(s_hp) >= 2:
                new_hp = int(s_hp[:2])
                if 10 <= new_hp <= 90:
                    f_hp = new_hp
            
            if f_hp > 90: f_hp = 0 
    except: 
        f_hp = data_final.get("horse_power")
    
    # Calculate costs
    elapsed = round(time.perf_counter() - start_time, 2)
    t4_cost_per_sec = 0.00006  # Estimated T4 cloud cost
    job_cost = round(elapsed * t4_cost_per_sec, 6)
    
    try: 
        if "confidence_score" in data_final:
            final_conf = float(data_final["confidence_score"])
        else:
            final_conf = 1.0 # Implicitly confident if we skipped Pass 2
    except: final_conf = 1.0

    result = {
        "doc_id": root,
        "fields": {
            "dealer_name": data_final.get("dealer_name"),
            "model_name": f_model,
            "horse_power": f_hp,
            "asset_cost": data_final.get("asset_cost"),
             "signature": {"present": True if detections.get("signature") else False, "bbox": detections.get("signature")},
             "stamp": {"present": True if detections.get("stamp") else False, "bbox": detections.get("stamp")}
        },
        "confidence": final_conf,
        "processing_time_sec": elapsed,
        "cost_estimate_usd": job_cost
    }
    
    return result

def main():
    parser = argparse.ArgumentParser(description="Invoice Extraction Pipeline")
    parser.add_argument("input_path", help="Path to image file or directory")
    args = parser.parse_args()
    
    load_models()
    warmup_pipeline()
    
    input_path = args.input_path
    
    if os.path.isdir(input_path):
        results = []
        files = [f for f in os.listdir(input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf'))]
        files = sorted(files)
        
        for f in files:
            full_path = os.path.join(input_path, f)
            try:
                res = process_invoice(full_path)
                results.append(res)
            except Exception:
                results.append({"doc_id": f, "error": "Processing Failed"})
        
        # Save aggregated JSON
        out_file = os.path.join(BASE_OUTPUT_DIR, "result.json")
        with open(out_file, "w", encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        # Not printing big list to stdout to keep it clean, or could print summary?
        # User asked for "output JSON output only" but also "if you provide a folder .. it will output a JSON ... for all images"
        # Usually implies saving to file.
            
    else:
        # Single File Mode
        try:
            res = process_invoice(input_path)
            
            # --- APPEND LOGIC ---
            out_file = os.path.join(BASE_OUTPUT_DIR, "result.json")
            existing_data = []

            # 1. Try to read existing file
            if os.path.exists(out_file):
                try:
                    with open(out_file, "r", encoding='utf-8') as f:
                        content = json.load(f)
                        if isinstance(content, list):
                            existing_data = content
                        elif isinstance(content, dict):
                            # Convert legacy single-object to list
                            existing_data = [content]
                except:
                    # File corrupt or empty? Start new list.
                    existing_data = []
            
            # 2. Append new result
            existing_data.append(res)

            # 3. Write back atomic list
            with open(out_file, "w", encoding='utf-8') as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False)
                
            # Print single result to stdout (for piping/logging)
            print(json.dumps(res, indent=2, ensure_ascii=False))
            
        except Exception as e:
            sys.stderr.write(f"Error: {e}\n")

if __name__ == "__main__":
    main()
