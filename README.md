# LLaMA 2 from scratch
LLaMA 2 model implemented from the beginning.

This is not a production or a proper implementation. The purpose of this code is to self-educate, understand and explore the principles of the LLaMA 2 model.

Based on the excellent tutorials by Umar Jamil:
* [LLaMA explained: KV-Cache, Rotary Positional Embedding, RMS Norm, Grouped Query Attention, SwiGLU](https://youtu.be/Mn_9W1nCFLo?si=9A0K9djlJGSn3Rt3), 
* [Coding LLaMA 2 from scratch in PyTorch - KV Cache, Grouped Query Attention, Rotary PE, RMSNorm](https://youtu.be/oM4VmoabDAI?si=fzfzZvq9A9mq3bfn). 

Thank you very much for the excellent tutorial and all credit goes to @hkproj.

## Getting started
```
# Create your virtual env
python3 -m venv venv

# Activate the venv
source venv/bin/activate

# Install required packages
pip install -r requirements.txt

# Visit the Meta Llama website: https://www.llama.com/llama-downloads/

# Go through the wizard and get the link.

# Download the weights (tested with the 7B model)
cd ./weights && ./download.sh

#  Run the the LLaMA model
python3 -m llama2 (Note: use -m to switch between interactive (i) and batch (b) mode)
```