import os
import torch
import torch.nn as nn
from unet import UNet

# ----------------------------
# CONFIG
# ----------------------------
MODEL_PATH = "save/initial/test/initial-57/Finetuned.pth" 
CHANNEL_FILE = "save/initial/test/initial-57/pruned_channels.txt"  # Change this to compare different models
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

def analyze_checkpoint(checkpoint_path):
    """Analyze the checkpoint to determine model configuration"""
    print(f"\n=== Analyzing Checkpoint: {checkpoint_path} ===")
    
    device = torch.device("cpu")  # Load to CPU first for analysis
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Check if it's a state dict or has extra keys
    if isinstance(checkpoint, dict) and 'mask_values' in checkpoint:
        print("Checkpoint type: Dict with metadata")
        state_dict = {k: v for k, v in checkpoint.items() if k != 'mask_values'}
        mask_values = checkpoint.get('mask_values', [0, 1])
        print(f"Mask values: {mask_values}")
    else:
        print("Checkpoint type: Pure state dict")
        state_dict = checkpoint
    
    # Detect n_classes from output layer
    if 'outc.conv.weight' in state_dict:
        n_classes = state_dict['outc.conv.weight'].shape[0]
    elif 'outc.conv.weight' in checkpoint:
        n_classes = checkpoint['outc.conv.weight'].shape[0]
    else:
        n_classes = 1
        print("Warning: Could not detect n_classes, defaulting to 1")
    
    print(f"Detected n_classes: {n_classes}")
    
    # Check architecture naming convention
    has_double_conv = any('double_conv' in k for k in state_dict.keys())
    has_maxpool_conv = any('maxpool_conv' in k for k in state_dict.keys())
    
    if has_double_conv or has_maxpool_conv:
        print("Architecture: Pytorch-UNet style (double_conv, maxpool_conv)")
        arch_type = "pytorch-unet"
    else:
        print("Architecture: Pruning repo style (conv, mpconv)")
        arch_type = "pruning"
    
    return n_classes, state_dict, arch_type

def load_model_smart(model_path, channel_file=None):
    """Smart model loading that handles different checkpoint formats"""
    device = torch.device("cuda" if USE_GPU and torch.cuda.is_available() else "cpu")
    
    n_classes, state_dict, arch_type = analyze_checkpoint(model_path)
    
    print(f"\nLoading model to {device}...")
    
    # Try to load with detected n_classes
    try:
        if channel_file and os.path.exists(channel_file):
            model = UNet(n_channels=3, n_classes=n_classes, f_channels=channel_file)
        else:
            model = UNet(n_channels=3, n_classes=n_classes)
        
        model = model.to(device)
        model.load_state_dict(state_dict, strict=False)
        print("✓ Model loaded successfully (with warnings ignored)")
        
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        print("\nTrying alternative approach...")
        
        # Try without channel file
        model = UNet(n_channels=3, n_classes=n_classes)
        model = model.to(device)
        model.load_state_dict(state_dict, strict=False)
        print("✓ Model loaded with basic architecture")
    
    model.eval()
    return model, arch_type

# ----------------------------
# MAIN
# ----------------------------
def main():
    print("="*60)
    print("MODEL ARCHITECTURE INSPECTOR")
    print("="*60)
    
    # Check if channel file exists
    channel_file = "model_channels.txt" if os.path.exists("model_channels.txt") else None
    
    try:
        model, arch_type = load_model_smart(MODEL_PATH, CHANNEL_FILE)
        
        print("\n" + "="*60)
        print("MODEL SUMMARY")
        print("="*60)
        print(model)
        
        print_model_channels(model)
        count_params(model)
        
        print("\n" + "="*60)
        print(f"Architecture Type: {arch_type}")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()