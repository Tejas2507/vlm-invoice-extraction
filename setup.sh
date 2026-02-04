#!/bin/bash

echo "Starting strict installation..."
# 2. Install HuggingFace ecosystem
echo "Installing Transformers/PEFT..."
pip install transformers==4.57.6 datasets==4.3.0 peft==0.18.1

# 3. Install Quantization
echo "Installing BitsAndBytes/TRL..."
pip install bitsandbytes==0.49.1 trl==0.27.0

# 4. Install Utilities
echo "Installing Utilities..."
pip install accelerate==1.12.0 pillow==11.3.0 tqdm==4.67.1 RapidFuzz==3.14.3 ultralytics==8.4.6

# 5. Install Unsloth
echo "Installing Unsloth..."
pip install unsloth

# 6. Pre-download Model to Cache (Enables Offline Run)
echo "Downloading Model to Cache..."
python3 -c "from unsloth import FastVisionModel; FastVisionModel.from_pretrained('unsloth/Qwen3-VL-8B-Instruct-bnb-4bit', load_in_4bit=True)"

echo "Installation Complete!"
