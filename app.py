import streamlit as st
import os
import glob
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms, datasets
from torchvision.utils import make_grid, save_image
from skimage.metrics import structural_similarity as ssim
from skimage.exposure import match_histograms
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import cv2

# Enable OpenCV multithreading
cv2.setUseOptimized(True)
cv2.setNumThreads(os.cpu_count())

def load_image_folder(folder, size=64):
    """
    Recursively load all images (png/jpg/jpeg) from a folder and its subfolders,
    convert to grayscale tensors normalized to [-1,1] of given size.
    """
    pattern = os.path.join(folder, "**", "*")
    paths = sorted(glob.glob(pattern, recursive=True))
    imgs = []
    for p in paths:
        if os.path.isfile(p) and p.lower().endswith(('png', 'jpg', 'jpeg')):
            try:
                img = Image.open(p).convert('L').resize((size, size))
                arr = np.array(img).astype(np.float32)
                tensor = torch.from_numpy(arr / 127.5 - 1.0).unsqueeze(0)
                imgs.append(tensor)
            except Exception as e:
                st.warning(f"Could not load image {p}: {e}")
    return torch.stack(imgs) if imgs else None

# DCGAN Generator
nz, ngf, nc = 100, 64, 1
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf), nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )
    def forward(self, x):
        return self.model(x)

device = torch.device("cpu")

@st.cache_resource
def load_superres_models(sr4_path, sr2_path):
    rrdb4 = RRDBNet(3, 3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    sr4 = RealESRGANer(scale=4, model_path=sr4_path, model=rrdb4,
                       tile=0, tile_pad=10, pre_pad=0, half=False, device=device)
    rrdb2 = RRDBNet(3, 3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    sr2 = RealESRGANer(scale=2, model_path=sr2_path, model=rrdb2,
                       tile=0, tile_pad=10, pre_pad=0, half=False, device=device)
    return sr4, sr2

# Post-processing

def guided_bilateral(np_img, d=5, sigmaColor=50, sigmaSpace=50):
    return cv2.bilateralFilter(np_img, d, sigmaColor, sigmaSpace)

def sharp_filter(tensor_img):
    kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]], dtype=np.float32)
    np_im = (tensor_img.squeeze().cpu().numpy() * 127.5 + 127.5).astype(np.uint8)
    sharp = cv2.filter2D(np_im, -1, kernel)
    return torch.from_numpy(sharp.astype(np.float32) / 127.5 - 1.0).unsqueeze(0)

def denoise_nl(tensor_img, h=3):
    np_img = (tensor_img.squeeze().cpu().numpy() * 127.5 + 127.5).astype(np.uint8)
    dn = cv2.fastNlMeansDenoising(np_img, None, h, 7, 21)
    return torch.from_numpy(dn.astype(np.float32) / 127.5 - 1.0).unsqueeze(0)

def clahe(tensor_img, clipLimit=2.0, tileGrid=(8,8)):
    np_img = (tensor_img.squeeze().cpu().numpy() * 127.5 + 127.5).astype(np.uint8)
    c = cv2.createCLAHE(clipLimit, tileGrid).apply(np_img)
    return torch.from_numpy(c.astype(np.float32) / 127.5 - 1.0).unsqueeze(0)

def hist_match(tensor_img, ref_uint8):
    np_img = (tensor_img.squeeze().cpu().numpy() * 127.5 + 127.5).astype(np.uint8)
    m = match_histograms(np_img, ref_uint8, channel_axis=None)
    return torch.from_numpy(np.clip(m, 0, 255).astype(np.float32) / 127.5 - 1.0).unsqueeze(0)

# Helper for metrics

def to_uint8(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, -1.0, 1.0)
    x = ((x + 1.0) * 127.5).round().to(torch.uint8)
    return x

# Metrics: IS, FID, SSIM

def compute_metrics(real_imgs, fake_imgs):
    real_u8 = to_uint8(real_imgs)
    fake_u8 = to_uint8(fake_imgs)
    # replicate channels
    if real_u8.shape[1] == 1:
        real_u8 = real_u8.repeat(1, 3, 1, 1)
    if fake_u8.shape[1] == 1:
        fake_u8 = fake_u8.repeat(1, 3, 1, 1)

    is_val, is_std = InceptionScore()(fake_u8.to(device))

    fid = FrechetInceptionDistance()
    fid.update(real_u8.to(device), real=True)
    fid.update(fake_u8.to(device), real=False)
    fid_val = fid.compute()

    real_np = real_imgs.squeeze().cpu().numpy()
    fake_np = fake_imgs.squeeze().cpu().numpy()
    ssim_vals = [ssim(r, f, data_range=2.0) for r, f in zip(real_np, fake_np)]

    return float(is_val), float(is_std), float(fid_val), float(np.mean(ssim_vals))

# Binary AUC discriminator

def compute_auc(real_dir, fake_imgs):
    tf = transforms.Compose([transforms.Resize(64), transforms.Grayscale(), transforms.ToTensor()])
    ds = datasets.ImageFolder(real_dir, transform=tf)
    loader = torch.utils.data.DataLoader(ds, batch_size=32)
    Xr = [imgs.view(imgs.size(0), -1).numpy() for imgs, _ in loader]
    Xr = np.vstack(Xr)
    Xf = fake_imgs.view(fake_imgs.size(0), -1).cpu().numpy()

    X = np.vstack([Xr, Xf])
    y = np.hstack([np.ones(Xr.shape[0]), np.zeros(Xf.shape[0])])

    clf = LogisticRegression(max_iter=500).fit(X, y)
    prob = clf.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, prob)
    fpr, tpr, _ = roc_curve(y, prob)
    return auc, fpr, tpr

# Streamlit UI
st.set_page_config(layout="wide", page_title="Synthetic CT Portal")
st.title("🧠 Synthetic CT Generator & Analyzer")

tab_gen, tab_analyze = st.tabs(["Generate CTs", "Analyze CTs"])

with tab_gen:
    st.header("Generate Synthetic CTs")
    ct_type = st.selectbox("CT Type", ["Normal Brain CT", "ICH Brain CT"])
    ckpt = st.file_uploader("Generator Checkpoint (pth/pt)")
    num_gen = st.slider("Number of CTs", 1, 400, 100)
    post = st.checkbox("Post-Process", True)
    sr = st.checkbox("Super-Resolution", False)

    if st.button("Generate & Save"):
        if not ckpt:
            st.error("Upload checkpoint first.")
        else:
            G = Generator().to(device)
            G.load_state_dict(torch.load(ckpt, map_location=device))
            base_out = os.path.join("./Dataset (Synthetic)", ct_type.replace(' ', '_'))
            os.makedirs(base_out, exist_ok=True)
            idx = len([d for d in os.listdir(base_out) if d.startswith('generation_')]) + 1
            out_dir = os.path.join(base_out, f'generation_{idx}')
            os.makedirs(out_dir, exist_ok=True)

            noise = torch.randn(num_gen, nz, 1, 1, device=device)
            raw_imgs = G(noise).cpu()
            proc_imgs = []
            for i, img in enumerate(raw_imgs):
                x = img.unsqueeze(0)
                if post:
                    ui8 = (x.squeeze().numpy() * 127.5 + 127.5).astype(np.uint8)
                    x = torch.from_numpy(guided_bilateral(ui8)).unsqueeze(0)
                    x = sharp_filter(x)
                    x = denoise_nl(x)
                    x = clahe(x)
                proc_imgs.append(x.squeeze(0))
                save_image(x, os.path.join(out_dir, f'synth_{i:03d}.png'), normalize=True)

            st.success(f"Saved {len(proc_imgs)} images to {out_dir}")
            grid = make_grid(torch.stack(proc_imgs), nrow=10, normalize=True)
            img_np = (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            st.image(img_np, use_container_width=True)

with tab_analyze:
    st.header("Analyze CT Sets")
    real_dir = st.selectbox("Real CT Set", sorted(os.listdir("./Dataset (East Cyprus Hospital)")))
    synth_set = st.selectbox("Synthetic Type", sorted(os.listdir("./Dataset (Synthetic)")))
    generation = st.selectbox("Generation", sorted(os.listdir(os.path.join("./Dataset (Synthetic)", synth_set))))

    if st.button("Load & Evaluate"):
        rpath = os.path.join("./Dataset (East Cyprus Hospital)", real_dir)
        spath = os.path.join("./Dataset (Synthetic)", synth_set, generation)
        real = load_image_folder(rpath)
        fake = load_image_folder(spath)
        if real is None or fake is None:
            st.error("Failed loading images.")
        else:
            st.subheader("Synthetic Samples Preview")
            grid2 = make_grid(fake, nrow=10, normalize=True)
            img2 = (grid2.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            st.image(img2, use_container_width=True)

            is_v, is_s, fid_v, ssim_v = compute_metrics(real, fake)
            auc, fpr, tpr = compute_auc(rpath, fake)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Inception Score", f"{is_v:.2f} ± {is_s:.2f}")
            c2.metric("FID", f"{fid_v:.2f}")
            c3.metric("SSIM", f"{ssim_v:.3f}")
            c4.metric("AUC", f"{auc:.3f}")

            df = pd.DataFrame({"FID": [fid_v], "SSIM": [ssim_v], "IS": [is_v]})
            st.bar_chart(df)

            fig = plt.figure()
            plt.plot(fpr, tpr, label='Real vs Synth')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('FPR')
            plt.ylabel('TPR')
            plt.title('ROC Curve')
            plt.legend()
            st.pyplot(fig)
