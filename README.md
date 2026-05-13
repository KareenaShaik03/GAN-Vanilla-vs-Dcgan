# 🚀 GAN Showcase — From Vanilla GAN to DCGAN

This repository demonstrates my understanding of **Generative Adversarial Networks (GANs)** by implementing and comparing:

* 🎨 **Vanilla GAN (Fully Connected)**
* 🖼️ **DCGAN (Deep Convolutional GAN)**

Both models are trained on the **CelebA dataset** to generate human face images.

---

## 📌 Why This Project?

Instead of building just one GAN, this project focuses on:

* Understanding **core GAN fundamentals**
* Identifying **limitations of Vanilla GAN**
* Improving performance using **DCGAN architecture**
* Comparing results in terms of **image quality and stability**

---

## 🧠 Models Covered

### 🔹 Vanilla GAN

* Fully connected (MLP-based) architecture
* No spatial awareness
* Produces blurry images
* Good for learning GAN basics

📂 Explore: `vanilla-gan/`

---

### 🔹 DCGAN

* Convolutional architecture (Conv + ConvTranspose)
* Learns spatial features
* Produces sharper and more realistic images
* More stable training

📂 Explore: `dcgan/`

---

## ⚖️ Comparison

| Feature             | Vanilla GAN     | DCGAN         |
| ------------------- | --------------- | ------------- |
| Architecture        | Fully Connected | Convolutional |
| Spatial Awareness   | ❌               | ✅             |
| Image Quality       | Blurry          | Sharp         |
| Training Stability  | Low             | Higher        |
| Batch Normalization | ❌               | ✅             |

---

## 🖼️ Results

| Model       | Output Quality            |
| ----------- | ------------------------- |
| Vanilla GAN | Basic face-like patterns  |
| DCGAN       | Realistic face structures |

*(Add sample generated images here if possible)*

---

## 🛠️ Tech Stack

* Python
* PyTorch
* Jupyter Notebook
* Computer Vision (Image Generation)

---

## 🚀 How to Run

1. Clone the repo:

```bash
git clone https://github.com/your-username/GAN-Showcase.git
cd GAN-Showcase
```

2. Go to any model:

```bash
cd vanilla-gan
# or
cd dcgan
```

3. Install dependencies:

```bash
pip install torch torchvision matplotlib
```

4. Run the notebook

---

## 📈 Key Learnings

* GANs work as a **minimax game between Generator & Discriminator**
* Fully connected GANs fail for image data due to lack of spatial understanding
* Convolutional layers significantly improve **feature learning**
* Training stability is a major challenge in GANs

---

## 🔮 Future Improvements

* WGAN / WGAN-GP
* Conditional GAN (cGAN)
* StyleGAN exploration
* Hyperparameter tuning

---

## 👩‍💻 Author

**Kareena Shaik**
Aspiring AI/ML Engineer | GenAI Enthusiast

---
