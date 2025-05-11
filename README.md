# Synthetic-CT-ICH-Normal-Generation-without-GPU

A full pipeline for photo-realistic synthetic CT slice generation—no GPU required.  
Implements:

1. **DCGAN** on 64×64 ICH/normal brain CT patches  
2. **Post-processing** (guided bilateral filtering, sharp-kernel sharpening, non-local means denoising, CLAHE, histogram matching)  
3. **Two-stage Super-Resolution** (4× then 2×) with RealESRGAN  
4. **Kernel-based final sharpening** for 512×512 outputs  
5. **Interactive demos** in `app.py` / `app2.py` and a notebook (`Code.ipynb`)  

## 📁 Repository Structure
```
├── Code.ipynb
├── app.py                  # Flask demo: generate & display synthetic CT
├── app2.py                 # Alternative Flask UI (batch sampling)
├── flowchart.md            # Pipeline flowchart & block diagram
├── Reference CT – 1.png    # Example reference slice (histogram matching)
├── Reference CT – 2.jpg
├── Reference CT – 3.jpg
├── Reference CT – 4.jpg
├── Dataset/                # ── place your real CT slices here
│   └── ICH Brain CT/
│       ├── patient01.png
│       └── …
├── requirements.txt
└── README.md
```
## 🛠️ Requirements

Tested on CPU; GPU optional.
```python
torch>=1.12.0
torchvision>=0.13.0
basicsr>=1.3.4
realesrgan>=0.3.0
opencv-python>=4.5.5
scikit-image>=0.19.0
matplotlib>=3.4.3
Pillow>=9.0.1
flask>=2.0
```
## ⚙️ Configuration

### 1. Reference CT

Choose one representative CT slice (clean, high-contrast) and place it in the repo root:

![Ref Img](<Reference%20CT%20-%204.jpg>)

This image is used for histogram matching.

### 2. Dataset

Organize your real CT scans under:

```
Dataset/ICH Brain CT/
  ├── slice1.png
  ├── slice2.png
  └── …
```

Supported formats: PNG, JPG, etc. Converted to grayscale internally.

### 3. RealESRGAN Weights

Download pretrained weights into the repo root:

```bash
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2/RealESRGAN_x4plus.pth
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2/RealESRGAN_x2plus.pth
```

## 🚀 Usage

### A. Notebook (Code.ipynb)

1. Open `Code.ipynb` in your Jupyter environment.
2. Adjust paths & hyperparameters in the first cell:

   ```python
   device = torch.device("cpu")   # or "cuda:0"
   dataset_root = "Dataset/ICH Brain CT"
   ref_ct_path  = "Reference CT – 1.png"
   out_dir      = "output/"
   num_epochs   = 100
   batch_size   = 16
   ```
3. Run all cells.
4. Review intermediate grids and final SR outputs in `output/`.

### B. Streamlit Implementation (`app.py` / `app2.py`)

1. Launch the web app:

   ```bash
   streamlit run app.py   # or app2.py
   ```
2. Open the generated link in your browser.

![Streamlit Img](<Screenshot-streamlit.png>)

3. Click **“Generate”** to sample a batch of synthetic CT slices.
4. The app will run the full pipeline (GAN→post-proc→SR) on CPU.

## 🔍 Pipeline Overview

See [flowchart.md](flowchart.md) for a block-diagram of the end-to-end data flow:

1. **Data Loading & Preprocessing** (64×64 grayscale patches)
2. **DCGAN Training**
3. **Guided Bilateral Filtering** → extract high-freq noise map
4. **Sharpening → Denoising → CLAHE → Histogram Matching**
5. **RealESRGAN ×4 → RealESRGAN ×2**
6. **Noise-map Reinjection** (α = 0.9)
7. **Final 3×3 Kernel Sharpen**

## 🎯 Outputs

* **Epoch grids**:
  `output/epoch_<n>.png` (400 samples, 20×20 grid)

* **Final SR images**:
  `output/sr8_epoch<n>_img<i>.png` (512×512, real-look slices)

---

## 📝 Notes & Tips

* **CPU vs. GPU**: CPU only is slow; use `device = torch.device("cuda")` if available.
* **Hyperparameter Tuning**:

  * `h` in NL-Means Denoise
  * `clipLimit` & `tileGridSize` in CLAHE
  * `alpha` for noise reinjection
    
* **Histogram Matching**: Choose a clean, high-contrast reference slice for best results.

> *PKumar-2002 · 2025*

