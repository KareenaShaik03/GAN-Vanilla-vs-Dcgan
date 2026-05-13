# 🖼️ DCGAN — Deep Convolutional GAN for Face Generation

A PyTorch implementation of **DCGAN (Deep Convolutional GAN)** trained on the CelebA dataset, using transposed convolutions in the Generator and strided convolutions in the Discriminator to generate high-quality 64×64 face images.

---

## 📌 Project Overview

This project implements the DCGAN architecture from Radford et al. (2015), replacing the dense layers of vanilla GAN with convolutional layers. This results in spatially aware feature learning and significantly better image quality.

| Component | Details |
|---|---|
| Dataset | CelebA (202,599 face images) |
| Image Size | 64 × 64 × 3 (RGB) |
| Architecture | Deep Convolutional (DCGAN) |
| Noise Dim (z) | 100 |
| Framework | PyTorch |

---

## 🏗️ Architecture

### Generator (Transposed Convolutions — Upsample)
```
ConvTranspose2d(100 → 1024, k=4, s=1, p=0) → BatchNorm → ReLU   # 1×1 → 4×4
ConvTranspose2d(1024 → 512, k=4, s=2, p=1) → BatchNorm → ReLU   # 4×4 → 8×8
ConvTranspose2d(512 → 256,  k=4, s=2, p=1) → BatchNorm → ReLU   # 8×8 → 16×16
ConvTranspose2d(256 → 128,  k=4, s=2, p=1) → BatchNorm → ReLU   # 16×16 → 32×32
ConvTranspose2d(128 → 3,    k=4, s=2, p=1) → Tanh                # 32×32 → 64×64
```

### Discriminator (Strided Convolutions — Downsample)
```
Conv2d(3 → 128,    k=4, s=2, p=1) → LeakyReLU(0.2)              # 64×64 → 32×32
Conv2d(128 → 256,  k=4, s=2, p=1) → BatchNorm → LeakyReLU(0.2)  # 32×32 → 16×16
Conv2d(256 → 512,  k=4, s=2, p=1) → BatchNorm → LeakyReLU(0.2)  # 16×16 → 8×8
Conv2d(512 → 1024, k=4, s=2, p=1) → BatchNorm → LeakyReLU(0.2)  # 8×8 → 4×4
Conv2d(1024 → 1,   k=4, s=1, p=0) → Sigmoid                     # 4×4 → 1×1
```

---

## ⚙️ Training Details

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 0.0002 |
| Betas | (0.5, 0.999) |
| Batch Size | 128 |
| Epochs | 5 |
| Loss Function | Binary Cross Entropy (BCELoss) |

**Training Strategy:**
1. Sample random noise `z` of shape `(batch, 100, 1, 1)`
2. Train Discriminator on real and fake images
3. Train Generator to fool the Discriminator
4. D loss = (real_loss + fake_loss) / 2

---

## 🗂️ Project Structure

```
dcgan-celeba/
│
├── Dcgan.ipynb              # Main notebook
├── README.md
└── outputs/                 # Generated images per epoch
```

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install torch torchvision torchaudio pillow matplotlib
```

### Dataset
Download the **CelebA** dataset from [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and update the `root_dir_path` in the notebook:
```python
root_dir_path = "path/to/img_align_celeba"
```

### Run
Open and run `Dcgan.ipynb` cell by cell. Generated images are saved and displayed after each epoch.

---

## 📊 Image Preprocessing

```python
transforms.CenterCrop(175)         # Crop face region
transforms.Resize(64)              # Resize to 64×64
transforms.ToTensor()
transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))  # Normalize to [-1, 1]
```

---

## 🔍 Key Concepts

- **ConvTranspose2d**: Learned upsampling (opposite of convolution) in the Generator
- **Strided Conv2d**: Downsampling without pooling layers in the Discriminator
- **BatchNorm2d**: Stabilizes training by normalizing feature maps
- **LeakyReLU**: Prevents dead neurons in the Discriminator
- **Tanh output**: Keeps generated pixel values in [-1, 1]
- **`.detach()`**: Stops Generator gradients when training Discriminator

---

## ✅ DCGAN vs Vanilla GAN

| Feature | Vanilla GAN | DCGAN |
|---|---|---|
| Architecture | Fully Connected | Convolutional |
| Spatial Awareness | ❌ | ✅ |
| Image Quality | Blurry | Sharper |
| BatchNorm | ❌ | ✅ |
| Training Stability | Lower | Higher |

---

## 🛠️ Tech Stack

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)

---

## 👩‍💻 Author

**Kareena Shaik** — [GitHub](https://github.com/KareenaShaik03)
