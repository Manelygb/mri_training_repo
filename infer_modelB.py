import os
import torch
import numpy as np
from PIL import Image
from unet import UNet

# ----------------------------
# CONFIG
# ----------------------------
MODEL_PATH = "MODEL.pth"
CHANNEL_FILE = "model_channels.txt"   # REQUIRED
INPUT_DIR = "inf_test"
OUTPUT_DIR = "inf_out"

IMG_SCALE = 0.5
THRESHOLD = 0.3
USE_GPU = True

# ----------------------------
# DEVICE
# ----------------------------
device = torch.device(
    "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"
)

# ----------------------------
# LOAD MODEL (CHECKPOINT-ACCURATE)
# ----------------------------
print("Loading model...")

model = UNet(
    n_channels=3,
    n_classes=1,
    f_channels=CHANNEL_FILE
)

state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state)

model.to(device)
model.eval()

print("✓ Model loaded successfully")

# ----------------------------
# PREPROCESS
# ----------------------------
def preprocess_image(pil_img, scale):
    w, h = pil_img.size

    new_w = int(w * scale)
    new_h = int(h * scale)

    img = pil_img.resize((new_w, new_h), Image.BILINEAR)
    img = np.array(img, dtype=np.float32) / 255.0

    # HWC → CHW
    img = img.transpose(2, 0, 1)

    #img = torch.from_numpy(img).unsqueeze(0)
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

    return img, (w, h)

# ----------------------------
# POSTPROCESS
# ----------------------------
def postprocess_mask(prob_mask, orig_size):
    prob_mask = Image.fromarray(prob_mask)
    prob_mask = prob_mask.resize(orig_size, Image.BILINEAR)

    prob_mask = np.array(prob_mask)
    mask = (prob_mask > THRESHOLD).astype(np.uint8)

    return mask

# ----------------------------
# SAVE MASK
# ----------------------------
def save_mask(mask, path):
    img = Image.fromarray(mask * 255, mode="L")
    img.save(path)

# ----------------------------
# INFERENCE
# ----------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\nRunning inference...\n")

for fname in sorted(os.listdir(INPUT_DIR)):
    if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    img_path = os.path.join(INPUT_DIR, fname)
    out_path = os.path.join(
        OUTPUT_DIR,
        os.path.splitext(fname)[0] + "_mask.png"
    )

    pil_img = Image.open(img_path).convert("RGB")
    img_tensor, orig_size = preprocess_image(pil_img, IMG_SCALE)

    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        # IMPORTANT: model already outputs probabilities
        prob = model(img_tensor)

    prob = prob.squeeze().cpu().numpy()

    mask = postprocess_mask(prob, orig_size)
    save_mask(mask, out_path)

    print(f"✓ {fname} → {os.path.basename(out_path)}")

print("\nInference complete.")
print(f"Masks saved to: {OUTPUT_DIR}")
