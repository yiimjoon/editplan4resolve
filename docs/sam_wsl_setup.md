# WSL Setup: SAM3 + SAM Audio

This guide sets up the WSL environment used by VideoForge SAM subprocesses.

## 1) Create a venv
```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3.12-venv
python3 -m venv ~/vf-sam3
source ~/vf-sam3/bin/activate
python -m pip install -U pip
```

## 2) Install PyTorch CUDA (cu121)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Verify CUDA:
```bash
python - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY
```

## 3) HuggingFace login
```bash
pip install huggingface_hub
hf auth login
```

Make sure you have access to:
- `facebook/sam3-large`
- `facebook/sam-audio-large`

## 4) Install SAM3
```bash
cd ~
rm -rf sam3
git clone https://github.com/facebookresearch/sam3
cd sam3
pip install -e .
```

## 5) Install SAM Audio
```bash
cd ~
rm -rf sam-audio
git clone https://github.com/facebookresearch/sam-audio
cd sam-audio
pip install -e .
```

## 6) Sanity check
```bash
python - <<'PY'
import sam3, sam_audio, torch
print("sam3 ok:", sam3.__name__)
print("sam_audio ok:", sam_audio.__name__)
print("cuda:", torch.cuda.is_available())
PY
```

## 7) VideoForge settings
In the Settings tab:
- WSL distro: `Ubuntu`
- WSL python: `python3`
- WSL venv: `/home/<user>/vf-sam3`

