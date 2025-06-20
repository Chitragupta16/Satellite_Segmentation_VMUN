## Introduction

VM-UNet is a hybrid deep-learning architecture that combines the global context‐capturing power of Vision Transformers (ViTs) with the precise localization capabilities of UNet. In the Omdena Frankfurt project, VM-UNet helped segment satellite imagery to accurately map urban green spaces by understanding both fine details (like small tree clusters) and broader spatial patterns (like park boundaries).

---

## Setting Up the VM-UNet Environment

Follow these steps to set up the `vmunet` environment.

### 1️. Create & Activate Environment
```bash
conda create -n vmunet python=3.8 -y
conda activate vmunet
```

### 2️. Install CUDA & PyTorch
```bash
conda install -c nvidia/label/cuda-11.8.0 cuda-toolkit

pip install torch==1.13.0 \
            torchvision==0.14.0 \
            torchaudio==0.13.0 \
            --extra-index-url https://download.pytorch.org/whl/cu117
```

### 3️. Install Dependencies
```bash
pip install packaging timm==0.4.12 pytest chardet yacs termcolor submitit tensorboardX
pip install triton==2.0.0 causal_conv1d==1.0.0 mamba_ssm==1.0.1
pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs
```

### 4️. Update Dependencies
```bash
conda env update --file environment.yaml --update-deps
```



---

##  Training the VM-UNet Architecture

Before you start training, download the pre-trained weights and place them in a folder named `pre_trained_weights` at the project root.

```bash
# 1. Clone the repository
git clone https://github.com/f-kuzey-edes-huyal/VM-UNet_Satellite_Segmentation.git
```
## 2. Initialize conda (for WSL)
```bash
conda init
exec $SHELL
conda activate vmunet
```

## 3. Navigate to the project directory
```bash
cd /mnt/c/VM-UNet_Satellite-Segmentation
```
## 4. Ensure the library path includes WSL libs
```bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
```

