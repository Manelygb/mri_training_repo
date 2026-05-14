import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms

from unet import UNet


# ----------------------------
# CONFIG
# ----------------------------
MODEL_PATH = "unet_carvana_scale0.5_epoch2.pth"
CHANNEL_FILE = "model_channels.txt"
INPUT_DIR = "inf_test"
OUTPUT_DIR = "inf_out"
USE_GPU = True


# ----------------------------
# UTILITIES
# ----------------------------
def print_model_channels(model):
    print("\n=== Layer-wise Conv Channels ===")
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            print(
                f"{name:40s} "
                f"in={m.in_channels:<4d} "
                f"out={m.out_channels:<4d} "
                f"kernel={m.kernel_size}"
            )


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\n=== Parameter Count ===")
    print(f"Total parameters     : {total:,}")
    print(f"Trainable parameters : {trainable:,}")


def load_image(path):
    img = Image.open(path).convert("RGB")   # PIL image
    img = np.asarray(img, dtype=np.float32)  # HWC float32
    img /= 255.0                             # normalize to [0,1]

    # IMPORTANT: do NOT use torch.from_numpy here
    img = torch.tensor(img, dtype=torch.float32)  # SAFE conversion
    img = img.permute(2, 0, 1)                # HWC → CHW
    img = img.unsqueeze(0)                    # add batch dim

    return img

def save_mask(pred, path):
    """
    pred: Tensor [1, 1, H, W]
    """
    # Ensure CPU
    pred = pred.detach().cpu()

    # Apply sigmoid if model output is raw logits
    pred = torch.sigmoid(pred)

    # Extract single mask [H, W]
    mask = pred[0, 0].numpy()

    # Threshold → binary mask
    mask = (mask > 0.5).astype(np.uint8) * 255

    # Save
    Image.fromarray(mask, mode="L").save(path)


# ----------------------------
# MAIN
# ----------------------------
def main():
    device = torch.device("cuda" if USE_GPU and torch.cuda.is_available() else "cpu")

    print("Loading model...")
    model = UNet(
        n_channels=3,
        n_classes=1,
        f_channels=CHANNEL_FILE
    ).to(device)

    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()

    print("\n=== Model Summary ===")
    print(model)

    print_model_channels(model)
    count_params(model)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n=== Running inference ===")
    with torch.no_grad():
        for fname in sorted(os.listdir(INPUT_DIR)):
            if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            img_path = os.path.join(INPUT_DIR, fname)
            out_path = os.path.join(OUTPUT_DIR, fname)

            img = load_image(img_path).to(device)
            pred = model(img)
            save_mask(pred, out_path)

            print(f"Processed: {fname}")

    print("\nInference complete.")
    print(f"Outputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
