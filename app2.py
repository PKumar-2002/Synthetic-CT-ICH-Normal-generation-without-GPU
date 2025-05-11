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

# Device and threading
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cv2.setUseOptimized(True)
cv2.setNumThreads(os.cpu_count())

@st.cache_resource
def load_superres_models(sr4_path, sr2_path):
    rrdb4 = RRDBNet(3, 3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    sr4 = RealESRGANer(scale=4, model_path=sr4_path, model=rrdb4,
                       tile=0, tile_pad=10, pre_pad=0, half=False, device=device)
    rrdb2 = RRDBNet(3, 3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    sr2 = RealESRGANer(scale=2, model_path=sr2_path, model=rrdb2,
                       tile=0, tile_pad=10, pre_pad=0, half=False, device=device)
    return sr4, sr2

# Load images as tensors

def load_image_folder(folder, size=64):
    pattern = os.path.join(folder, "**", "*")
    paths = sorted(glob.glob(pattern, recursive=True))
    imgs = []
    for p in paths:
        if os.path.isfile(p) and p.lower().endswith(('png','jpg','jpeg')):
            try:
                img = Image.open(p).convert('L').resize((size, size))
                arr = np.array(img, dtype=np.float32)
                tensor = torch.from_numpy(arr / 127.5 - 1.0).unsqueeze(0)
                imgs.append(tensor)
            except Exception:
                pass
    return torch.stack(imgs) if imgs else None

# DCGAN generator
nz, ngf, nc = 100, 64, 1
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias=False), nn.BatchNorm2d(ngf*8), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf*4), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf*2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*2, ngf,   4, 2, 1, bias=False), nn.BatchNorm2d(ngf),   nn.ReLU(True),
            nn.ConvTranspose2d(ngf,   nc,    4, 2, 1, bias=False), nn.Tanh()
        )
    def forward(self, x):
        return self.model(x)

# Convert float [-1,1] to uint8

def to_uint8(x):
    x = torch.clamp(x, -1.0, 1.0)
    x = ((x + 1.0) * 127.5).round().to(torch.uint8)
    return x

# Compute IS, FID, SSIM in batches to save memory

def compute_metrics(real, fake, batch_size=32):
    real_u = to_uint8(real)
    fake_u = to_uint8(fake)
    # replicate channels
    if real_u.shape[1] == 1: real_u = real_u.repeat(1,3,1,1)
    if fake_u.shape[1] == 1: fake_u = fake_u.repeat(1,3,1,1)

    # Inception Score
    metric_is = InceptionScore()
    for i in range(0, fake_u.size(0), batch_size):
        chunk = fake_u[i:i+batch_size].to(device)
        metric_is.update(chunk)
    is_v, is_s = metric_is.compute()

    # FID
    metric_fid = FrechetInceptionDistance()
    for i in range(0, real_u.size(0), batch_size):
        metric_fid.update(real_u[i:i+batch_size].to(device), real=True)
    for i in range(0, fake_u.size(0), batch_size):
        metric_fid.update(fake_u[i:i+batch_size].to(device), real=False)
    fid_v = metric_fid.compute()

    # SSIM
    r_np = real.squeeze().cpu().numpy()
    f_np = fake.squeeze().cpu().numpy()
    ssim_vals = [ssim(r, f, data_range=2.0) for r, f in zip(r_np, f_np)]
    ssim_v = float(np.mean(ssim_vals))

    return float(is_v), float(is_s), float(fid_v), ssim_v

# Discriminator AUC

def compute_auc(real_dir, fake, batch_size=32):
    tf = transforms.Compose([transforms.Resize(64), transforms.Grayscale(), transforms.ToTensor()])
    ds = datasets.ImageFolder(real_dir, transform=tf)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size)
    Xr = []
    for imgs, _ in loader:
        Xr.append(imgs.view(imgs.size(0),-1).numpy())
    Xr = np.vstack(Xr)
    Xf = fake.view(fake.size(0),-1).cpu().numpy()
    X = np.vstack([Xr, Xf])
    y = np.hstack([np.ones(len(Xr)), np.zeros(len(Xf))])
    clf = LogisticRegression(max_iter=500).fit(X, y)
    prob = clf.predict_proba(X)[:,1]
    auc_v = roc_auc_score(y, prob)
    fpr, tpr, _ = roc_curve(y, prob)
    return auc_v, fpr, tpr

# Streamlit UI setup
st.set_page_config(layout="wide", page_title="Synthetic CT Portal")
st.title("🧠 Synthetic CT Generator & Analyzer")

tab_gen, tab_analyze = st.tabs(["Generate CTs", "Analyze CTs"])

with tab_gen:
    st.header("Generate Synthetic CTs")
    ct_type = st.selectbox("CT Type", ["Normal Brain CT", "ICH Brain CT"])
    ckpt = st.file_uploader("Generator Checkpoint (.pth/.pt)")
    num_gen = st.slider("Number of CTs", 1, 400, 100)
    post = st.checkbox("Post-Process", True)
    use_sr = st.checkbox("Super-Resolution", False)
    if use_sr:
        sr4_path = st.file_uploader("ESRGAN x4 model (.pth)")
        sr2_path = st.file_uploader("ESRGAN x2 model (.pth)")
    if st.button("Generate & Save"):
        if not ckpt:
            st.error("Upload generator checkpoint first.")
        else:
            G = Generator().to(device)
            G.load_state_dict(torch.load(ckpt, map_location=device))
            if use_sr and sr4_path and sr2_path:
                sr4, sr2 = load_superres_models(sr4_path, sr2_path)

            base_out = os.path.join("./Dataset (Synthetic)", ct_type.replace(' ', '_'))
            os.makedirs(base_out, exist_ok=True)
            idx = len([d for d in os.listdir(base_out) if d.startswith('generation_')]) + 1
            out_dir = os.path.join(base_out, f'generation_{idx}')
            os.makedirs(out_dir, exist_ok=True)

            noise = torch.randn(num_gen, nz, 1, 1, device=device)
            raw = G(noise).cpu()
            proc = []
            for i, img in enumerate(raw):
                x = img.unsqueeze(0)
                if post:
                    ui = (x.squeeze().numpy() * 127.5 + 127.5).astype(np.uint8)
                    x = torch.from_numpy(cv2.bilateralFilter(ui, 5, 50, 50)).unsqueeze(0)
                if use_sr:
                    x_img = (x.squeeze().permute(1,2,0).numpy()).astype(np.uint8)
                    x_sr, _ = sr4.enhance(x_img, outscale=4)
                    x = torch.from_numpy(x_sr.astype(np.float32)/127.5 - 1.0).permute(2,0,1).unsqueeze(0)
                proc.append(x.squeeze(0))
                save_image(x, os.path.join(out_dir, f'synth_{i:03d}.png'), normalize=True)

            st.success(f"Saved to {out_dir}")
            grid = make_grid(torch.stack(proc), nrow=10, normalize=True)
            st.image((grid.permute(1,2,0).numpy()*255).astype(np.uint8), use_container_width=True)

with tab_analyze:
    st.header("Analyze CT Sets")
    real_set = st.selectbox("Real CT Set", sorted(os.listdir("./Dataset (East Cyprus Hospital)")))
    synth_set = st.selectbox("Synthetic Type", sorted(os.listdir("./Dataset (Synthetic)")))
    analyze_all = st.checkbox("Analyze All Generations")
    if not analyze_all:
        generation = st.selectbox("Generation", sorted(os.listdir(os.path.join("./Dataset (Synthetic)" , synth_set))))
    if st.button("Evaluate"):
        real_dir = os.path.join("./Dataset (East Cyprus Hospital)", real_set)
        synth_base = os.path.join("./Dataset (Synthetic)", synth_set)

        if analyze_all:
            gens = sorted(os.listdir(synth_base))
            records = []
            for g in gens:
                fp = os.path.join(synth_base, g)
                real_imgs = load_image_folder(real_dir)
                fake_imgs = load_image_folder(fp)
                if real_imgs is not None and fake_imgs is not None:
                    iv, istd, fv, ss = compute_metrics(real_imgs, fake_imgs)
                    aucv, _, _ = compute_auc(real_dir, fake_imgs)
                    records.append({'Gen': g, 'IS': iv, 'FID': fv, 'SSIM': ss, 'AUC': aucv})
            df_all = pd.DataFrame(records).set_index('Gen')
            st.subheader("Metrics per Generation")
            st.dataframe(df_all)
            st.line_chart(df_all)
        else:
            fp = os.path.join(synth_base, generation)
            real_imgs = load_image_folder(real_dir)
            fake_imgs = load_image_folder(fp)
            st.subheader("Sample Synthetic Grid")
            grid2 = make_grid(fake_imgs, nrow=10, normalize=True)
            st.image((grid2.permute(1,2,0).numpy()*255).astype(np.uint8), use_container_width=True)

            iv, istd, fv, ss = compute_metrics(real_imgs, fake_imgs)
            aucv, fpr, tpr = compute_auc(real_dir, fake_imgs)

            df_metrics = pd.DataFrame({'Metric':['IS','FID','SSIM','AUC'], 'Value':[iv,fv,ss,aucv]}).set_index('Metric')
            st.subheader("Metrics")
            st.bar_chart(df_metrics)

            fig = plt.figure()
            plt.plot(fpr, tpr, label='ROC Curve')
            plt.plot([0,1],[0,1],'k--')
            plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
            plt.legend(); st.pyplot(fig)
