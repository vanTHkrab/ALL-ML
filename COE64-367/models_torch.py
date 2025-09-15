import os, glob, math, random
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from torchvision.models import EfficientNet_B0_Weights

# ----------------------------
# Config
# ----------------------------
IMG_DIR      = "./data/prepared-224-112"                 # <<< แก้เป็นที่เก็บรูป
CSV_PATH     = "./data/mangosteen-train-3.csv"        # ใช้ไฟล์ที่คุณอัป
IMG_SIZE     = 224
BATCH_SIZE   = 12
EPOCHS       = 1000
LR           = 3e-4
VAL_RATIO    = 0.2
SEED         = 42
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS  = 2 if DEVICE == "cuda" else 0
PIN_MEMORY   = DEVICE == "cuda"

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ----------------------------
# CSV utilities (robust column mapping)
# ----------------------------
def pick_first_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Cannot find any of columns: {candidates}")

def load_table(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # 1) map: id / weight_g / volume_ml (เพิ่มชื่อคอลัมน์ที่เป็นไปได้)
    id_col      = pick_first_col(df, ["A", "ID", "Id", "id"])
    weight_col  = pick_first_col(df, ["B", "Weight (g)", "weight_g", "weight", "น้ำหนัก", "น้ำหนัก (g)"])
    volume_col  = pick_first_col(df, ["D", "ปริมาตร", "volumes_ml", "volumes (ml)", "volume", "ปริมาตร (ml)"])

    ids = df[id_col].astype(str).str.strip()

    # 2) แปลง id ถ้าเป็นตัวเลขล้วน -> i%03d
    if ids.str.fullmatch(r"\d+").all():
        ids = ids.astype(int).map(lambda x: f"i{x:03d}")
    else:
        # ตัดนามสกุลไฟล์ออก และดึง prefix รูปแบบ iXXX ถ้ามี
        ids = ids.str.replace(r"\.(jpg|jpeg|png)$", "", regex=True)
        extracted = ids.str.extract(r"(i\d{3})", expand=False)
        ids = extracted.fillna(ids)

    out = pd.DataFrame({
        "id": ids,
        "weight_g": pd.to_numeric(df[weight_col], errors="coerce"),
        "volume_ml": pd.to_numeric(df[volume_col], errors="coerce"),
    }).dropna()

    return out

# ----------------------------
# Dataset
# ----------------------------
class MangoDataset(Dataset):
    def __init__(self, table: pd.DataFrame, img_dir: str, transform=None):
        self.df = table.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

        # Pre-resolve best image path for each id
        self.paths = []
        for _, r in self.df.iterrows():
            pid = r["id"]
            patt = [
                f"{pid}_crop_group_*.jpg",
                f"{pid}_*.jpg",
                f"{pid}*.jpg",
                f"{pid}_*.png",
                f"{pid}*.png",
            ]
            matches = []
            for p in patt:
                matches += glob.glob(os.path.join(self.img_dir, p))
            if not matches:
                raise FileNotFoundError(f"No image found for id={pid} in {self.img_dir}")
            # choose a deterministic file (first sorted)
            self.paths.append(sorted(matches)[0])

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.paths[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        w = torch.tensor([row["weight_g"]], dtype=torch.float32)
        y = torch.tensor([row["volume_ml"]], dtype=torch.float32)
        return img, w, y, row["id"]

# ----------------------------
# Transforms (ImageNet norm)
# ----------------------------
weights_enum = EfficientNet_B0_Weights.DEFAULT
preprocess = weights_enum.transforms()  # includes resize, center-crop, ToTensor, Normalize
# If you prefer fixed resize instead of center-crop, use this:
# preprocess = transforms.Compose([
#     transforms.Resize((IMG_SIZE, IMG_SIZE)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
# ])

# ----------------------------
# Model
# ----------------------------
class MangoModel(nn.Module):
    def __init__(self, freeze_backbone: bool = True):
        super().__init__()
        base = models.efficientnet_b0(weights=weights_enum)
        # take 1280-d features
        in_feats = base.classifier[1].in_features
        base.classifier = nn.Identity()
        self.backbone = base

        # small head: concat [img_feat(1280) + weight(1)] -> 1
        self.head = nn.Sequential(
            nn.Linear(in_feats + 1, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            # keep BN affine params trainable can help
            for m in self.backbone.modules():
                if isinstance(m, nn.BatchNorm2d):
                    for p in m.parameters():
                        p.requires_grad = True

    def forward(self, x, w):
        f = self.backbone(x)        # (N, 1280)
        x = torch.cat([f, w], dim=1)
        y = self.head(x)
        return y

# ----------------------------
# Metrics
# ----------------------------
def metrics(y_true, y_pred):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    mae = float(torch.mean(torch.abs(y_true - y_pred)))
    rmse = float(torch.sqrt(torch.mean((y_true - y_pred) ** 2)))
    # R^2
    ss_res = float(torch.sum((y_true - y_pred) ** 2))
    ss_tot = float(torch.sum((y_true - torch.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / (ss_tot + 1e-8)
    return mae, rmse, r2

# ----------------------------
# Train
# ----------------------------
def main():
    print(f"Device: {DEVICE} | CUDA: {torch.cuda.is_available()}")
    df = load_table(CSV_PATH)
    print(f"Rows: {len(df)} | Columns: {df.columns.tolist()}")

    dataset = MangoDataset(df, IMG_DIR, preprocess)

    n_val = max(1, int(len(dataset) * VAL_RATIO))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(SEED))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    model = MangoModel(freeze_backbone=True).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scaler = GradScaler(enabled=(DEVICE == "cuda"))

    best_val = math.inf

    for epoch in range(1, EPOCHS + 1):
        # ---- train
        model.train()
        loss_acc = 0.0
        for imgs, w, y, _ in train_loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            w    = w.to(DEVICE, non_blocking=True)
            y    = y.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(DEVICE == "cuda")):
                pred = model(imgs, w)
                loss = criterion(pred, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_acc += float(loss.item()) * imgs.size(0)

        train_loss = loss_acc / len(train_loader.dataset)

        # ---- validate
        model.eval()
        with torch.no_grad():
            all_y, all_p = [], []
            for imgs, w, y, _ in val_loader:
                imgs, w, y = imgs.to(DEVICE), w.to(DEVICE), y.to(DEVICE)
                with autocast(enabled=(DEVICE == "cuda")):
                    p = model(imgs, w)
                all_y.append(y)
                all_p.append(p)
            ycat = torch.cat(all_y, 0)
            pcat = torch.cat(all_p, 0)
            mae, rmse, r2 = metrics(ycat, pcat)
            val_loss = float(torch.mean((ycat - pcat) ** 2))

        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"train MSE {train_loss:.3f} | "
              f"val MSE {val_loss:.3f} | MAE {mae:.2f} | RMSE {rmse:.2f} | R2 {r2:.3f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(),
                        "cfg": {"img_size": IMG_SIZE}}, "mango_volume_efficientnet_b0.pt")
            print("  ✔ Saved best checkpoint")

if __name__ == "__main__":
    main()
