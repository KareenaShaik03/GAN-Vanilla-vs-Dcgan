# 🎨 Vanilla GAN — CelebA Face Generation

A from-scratch implementation of a **Vanilla Generative Adversarial Network (GAN)** trained on the CelebA dataset to generate realistic human face images using fully connected (dense) layers in PyTorch.

---

## 📌 Project Overview

This project implements the original GAN architecture proposed by Ian Goodfellow et al. (2014). Both the Generator and Discriminator are built using **dense (fully connected) layers**, making this a foundational baseline before moving to convolutional variants.

| Component | Details |
|---|---|
| Dataset | CelebA (202,599 face images) |
| Image Size | 64 × 64 × 3 (RGB) |
| Architecture | Fully Connected (MLP) |
| Noise Dim (z) | 100 |
| Framework | PyTorch |

---

## 🏗️ Architecture

### Generator
```
Linear(100 → 256)  → ReLU
Linear(256 → 512)  → ReLU
Linear(512 → 1024) → ReLU
Linear(1024 → 64×64×3) → Tanh
Output reshaped → (batch, 3, 64, 64)
```

### Discriminator
```
Flatten → Linear(64×64×3 → 1024) → LeakyReLU(0.2)
Linear(1024 → 512) → LeakyReLU(0.2)
Linear(512 → 256)  → LeakyReLU(0.2)
Linear(256 → 1)    → Sigmoid
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
1. Train Discriminator on real and fake images
2. Train Generator to fool the Discriminator
3. Discriminator loss = (real_loss + fake_loss) / 2

---

## 🗂️ Project Structure

```
vanilla-gan-celeba/
│
├── vanilla_gan.ipynb        # Main notebook
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
Open and run `vanilla_gan.ipynb` cell by cell. Generated images are displayed after each epoch.

---

## 📊 Image Preprocessing

```python
transforms.CenterCrop(178)         # Crop face region
transforms.Resize(64)              # Resize to 64×64
transforms.ToTensor()
transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))  # Normalize to [-1, 1]
```

---

## 🔍 Key Concepts

- **Generator**: Maps random noise `z ~ N(0,1)` to fake images
- **Discriminator**: Classifies images as real (1) or fake (0)
- **Adversarial Training**: Both networks compete in a minimax game
- **Tanh output** in Generator keeps pixel values in [-1, 1], matching normalized real images
- **LeakyReLU** in Discriminator prevents dying gradients

---

## 📈 Limitations & Next Steps

Vanilla GANs with MLP layers tend to produce **blurry outputs** on image data. Key limitations:
- No spatial awareness (no convolutions)
- Training instability (mode collapse risk)

➡️ See [DCGAN implementation](../dcgan-celeba/) for the convolutional upgrade.

---

## 🛠️ Tech Stack

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)

---

## 👩‍💻 Author

**Kareena Shaik** — [GitHub](https://github.com/KareenaShaik03)
