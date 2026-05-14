import os
import torch
import numpy as np
from PIL import Image
from unet import UNet

# ----------------------------
# CONFIG
# ----------------------------
MODEL_PATH = "MODEL.pth"
CHANNEL_FILE = "model_channels.txt"
INPUT_DIR = "inf_test"
OUTPUT_DIR = "inf_out"
USE_GPU = True
SCALE_FACTOR = 0.5  # Match the training scale

# ----------------------------
# IMAGE PREPROCESSING (matching original repo logic)
# ----------------------------
def resize_and_crop(img, scale=0.5):
    """Resize image maintaining aspect ratio"""
    w, h = img.size
    newW = int(w * scale)
    newH = int(h * scale)
    
    if newW % 1 != 0 or newH % 1 != 0:
        newW = int(newW)
        newH = int(newH)
    
    img = img.resize((newW, newH), Image.BILINEAR)
    return img

def normalize(img):
    """Normalize image to [0, 1]"""
    img = np.array(img)
    img = img.astype(np.float32) / 255.0
    return img

def split_img_into_squares(img):
    """Split image into left and right squares for processing"""
    # img should be HWC numpy array
    h, w, c = img.shape
    
    # Create square crops from left and right
    if w > h:
        # Landscape: take left and right squares
        left = img[:, :h, :]
        right = img[:, w-h:w, :]
    else:
        # Portrait or square: just use the image twice or crop
        left = img
        right = img
    
    return left, right

def merge_masks(left_mask, right_mask, original_width):
    """Merge left and right mask predictions"""
    h, w = left_mask.shape
    
    # Calculate how to merge based on original width
    if original_width <= h:
        # If original was square or portrait, just return left
        return left_mask
    
    # Create output mask with original aspect ratio
    full_w = int(original_width / (original_width / w))  # Adjust based on scaling
    
    # Create merged mask
    merged = np.zeros((h, full_w), dtype=np.float32)
    
    # Simple blending: left half from left_mask, right half from right_mask
    mid = full_w // 2
    merged[:, :mid] = left_mask[:, :mid] if left_mask.shape[1] >= mid else left_mask
    merged[:, mid:] = right_mask[:, -(full_w-mid):] if right_mask.shape[1] >= (full_w-mid) else right_mask
    
    return merged

def predict_img(net, full_img, scale_factor=0.5, out_threshold=0.5, use_gpu=False):
    """
    Predict mask for a full image
    """
    net.eval()
    
    # Store original dimensions
    orig_w, orig_h = full_img.size
    
    # Preprocess
    img = resize_and_crop(full_img, scale=scale_factor)
    img = normalize(img)
    
    # Split into squares
    left_square, right_square = split_img_into_squares(img)
    
    # Convert to tensors (CHW format) - using torch.tensor for compatibility
    left_chw = left_square.transpose(2, 0, 1)
    right_chw = right_square.transpose(2, 0, 1)
    
    X_left = torch.tensor(left_chw, dtype=torch.float32).unsqueeze(0)
    X_right = torch.tensor(right_chw, dtype=torch.float32).unsqueeze(0)
    
    if use_gpu:
        X_left = X_left.cuda()
        X_right = X_right.cuda()
    
    # Inference
    with torch.no_grad():
        output_left = torch.sigmoid(net(X_left))
        output_right = torch.sigmoid(net(X_right))
        
        # Get probabilities
        left_probs = output_left.squeeze().cpu().numpy()
        right_probs = output_right.squeeze().cpu().numpy()
    
    # Merge predictions
    # For simplicity, if image is wide, merge; otherwise use left only
    scaled_w = int(orig_w * scale_factor)
    scaled_h = int(orig_h * scale_factor)
    
    if scaled_w > scaled_h:
        # Need to merge
        full_mask = merge_masks(left_probs, right_probs, scaled_w)
    else:
        full_mask = left_probs
    
    # Resize mask back to original size
    full_mask = Image.fromarray((full_mask * 255).astype(np.uint8))
    full_mask = full_mask.resize((orig_w, orig_h), Image.NEAREST)
    full_mask = np.array(full_mask) / 255.0
    
    # Threshold
    return (full_mask > out_threshold).astype(np.uint8)

def save_mask_as_png(mask, path):
    """
    Save binary mask as PNG
    mask: numpy array [H, W] with values 0 or 1
    """
    # Convert to 0-255
    mask_img = (mask * 255).astype(np.uint8)
    img = Image.fromarray(mask_img, mode='L')
    img.save(path)

# ----------------------------
# MAIN
# ----------------------------
def main():
    device = torch.device("cuda" if USE_GPU and torch.cuda.is_available() else "cpu")
    
    print(f"Loading model from {MODEL_PATH}...")
    model = UNet(
        n_channels=3,
        n_classes=1,
        f_channels=CHANNEL_FILE
    ).to(device)
    
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print("Model loaded successfully!")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"\nProcessing images from {INPUT_DIR}...")
    print("-" * 50)
    
    processed = 0
    for fname in sorted(os.listdir(INPUT_DIR)):
        if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        
        img_path = os.path.join(INPUT_DIR, fname)
        
        # Change extension to .png for output
        base_name = os.path.splitext(fname)[0]
        out_path = os.path.join(OUTPUT_DIR, f"{base_name}_OUT.png")
        
        # Load image
        img = Image.open(img_path).convert("RGB")
        
        # Predict
        mask = predict_img(
            net=model,
            full_img=img,
            scale_factor=SCALE_FACTOR,
            out_threshold=0.5,
            use_gpu=USE_GPU
        )
        
        # Save as PNG
        save_mask_as_png(mask, out_path)
        
        processed += 1
        print(f"✓ {fname:40s} -> {os.path.basename(out_path)}")
    
    print("-" * 50)
    print(f"\nInference complete!")
    print(f"Processed {processed} images")
    print(f"Masks saved to: {OUTPUT_DIR}")
    print(f"Format: PNG (grayscale, 0=background, 255=foreground)")

if __name__ == "__main__":
    main()