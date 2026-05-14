import torch
import torch.nn as nn
import os
import json
from optparse import OptionParser
from unet import UNet
from flops_counter import flops_count


def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_model_size(model, path=None):
    """Get model size in MB"""
    if path and os.path.exists(path):
        # Size on disk
        size_mb = os.path.getsize(path) / (1024 ** 2)
    else:
        # Size in memory
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        size_mb = (param_size + buffer_size) / (1024 ** 2)
    return size_mb


def get_layer_info(model):
    """Get detailed layer information"""
    layer_info = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            if isinstance(module, nn.Conv2d):
                layer_type = "Conv2d"
                in_ch, out_ch = module.weight.shape[1], module.weight.shape[0]
            else:
                layer_type = "ConvTranspose2d"
                in_ch, out_ch = module.weight.shape[0], module.weight.shape[1]
            
            params = module.weight.numel()
            if module.bias is not None:
                params += module.bias.numel()
            
            layer_info.append({
                'name': name,
                'type': layer_type,
                'in_channels': in_ch,
                'out_channels': out_ch,
                'kernel_size': module.kernel_size,
                'params': params
            })
    return layer_info


def calculate_flops(model, input_shape=(1, 3, 572, 572), gpu=False):
    """Calculate FLOPs for the model"""
    device = torch.device('cuda' if gpu and torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(input_shape).to(device)
    
    # Get FLOPs
    with torch.no_grad():
        flops = flops_count(model, input_shape[2:])  # Pass (H, W)
    
    return flops


def print_statistics(model_path, config_path=None, input_shape=(1, 3, 572, 572), gpu=False):
    """Print comprehensive model statistics"""
    device = torch.device('cuda' if gpu and torch.cuda.is_available() else 'cpu')
    
    # Load model
    print(f"\n{'='*80}")
    print(f"Loading model from: {model_path}")
    print(f"{'='*80}\n")
    
    # Initialize model
    net = UNet(n_channels=3, n_classes=2, bilinear=False)
    
    # Load state dict
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    net.load_state_dict(state_dict)
    net.to(device)
    net.eval()
    
    # Load config if available
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Pruning configuration loaded from: {config_path}")
        print(f"  Prune mode: {config.get('prune_mode', 'N/A')}")
        print(f"  Encoder ratio: {config.get('encoder_ratio', 'N/A')}")
        print(f"  Decoder ratio: {config.get('decoder_ratio', 'N/A')}")
        print(f"  Bottleneck ratio: {config.get('bottleneck_ratio', 'N/A')}\n")
    
    # Get statistics
    total_params, trainable_params = count_parameters(net)
    model_size = get_model_size(net, model_path)
    layer_info = get_layer_info(net)
    
    # Print basic statistics
    print(f"Model Statistics:")
    print(f"{'─'*80}")
    print(f"  Total parameters:      {total_params:,}")
    print(f"  Trainable parameters:  {trainable_params:,}")
    print(f"  Model size (disk):     {model_size:.2f} MB")
    print(f"  Model size (memory):   {get_model_size(net):.2f} MB")
    
    # Calculate FLOPs
    print(f"\n  Calculating FLOPs for input shape {input_shape}...")
    try:
        flops = calculate_flops(net, input_shape, gpu)
        if flops:
            total_flops = sum(flops)
            print(f"  Total FLOPs:           {total_flops:,}")
            print(f"  Total GFLOPs:          {total_flops / 1e9:.2f}")
    except Exception as e:
        print(f"  Could not calculate FLOPs: {e}")
    
    # Print layer-by-layer information
    print(f"\n{'─'*80}")
    print(f"Layer-by-Layer Information:")
    print(f"{'─'*80}")
    print(f"{'Layer Name':<30} {'Type':<15} {'In':<8} {'Out':<8} {'Params':>12}")
    print(f"{'─'*80}")
    
    for layer in layer_info:
        print(f"{layer['name']:<30} {layer['type']:<15} "
              f"{layer['in_channels']:<8} {layer['out_channels']:<8} "
              f"{layer['params']:>12,}")
    
    print(f"{'─'*80}\n")
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': model_size,
        'layer_info': layer_info,
        'flops': flops if 'flops' in locals() else None
    }


def compare_models(original_path, pruned_path, original_config=None, pruned_config=None, 
                   input_shape=(1, 3, 572, 572), gpu=False):
    """Compare original and pruned models"""
    print(f"\n{'='*80}")
    print(f"COMPARING MODELS")
    print(f"{'='*80}")
    
    print("\n[1] ORIGINAL MODEL")
    original_stats = print_statistics(original_path, original_config, input_shape, gpu)
    
    print("\n[2] PRUNED MODEL")
    pruned_stats = print_statistics(pruned_path, pruned_config, input_shape, gpu)
    
    # Calculate reductions
    print(f"\n{'='*80}")
    print(f"COMPARISON SUMMARY")
    print(f"{'='*80}\n")
    
    param_reduction = (1 - pruned_stats['total_params'] / original_stats['total_params']) * 100
    size_reduction = (1 - pruned_stats['model_size_mb'] / original_stats['model_size_mb']) * 100
    
    print(f"  Parameter reduction:   {param_reduction:.2f}% "
          f"({original_stats['total_params']:,} → {pruned_stats['total_params']:,})")
    print(f"  Model size reduction:  {size_reduction:.2f}% "
          f"({original_stats['model_size_mb']:.2f} MB → {pruned_stats['model_size_mb']:.2f} MB)")
    
    if original_stats['flops'] and pruned_stats['flops']:
        original_flops = sum(original_stats['flops'])
        pruned_flops = sum(pruned_stats['flops'])
        flops_reduction = (1 - pruned_flops / original_flops) * 100
        print(f"  FLOPs reduction:       {flops_reduction:.2f}% "
              f"({original_flops / 1e9:.2f} GFLOPs → {pruned_flops / 1e9:.2f} GFLOPs)")
    
    print(f"\n{'='*80}\n")


def get_args():
    parser = OptionParser()
    parser.add_option('-m', '--model', dest='model_path',
                      help='Path to model checkpoint (.pth file)')
    parser.add_option('-c', '--config', dest='config_path', default=None,
                      help='Path to pruned config JSON (optional)')
    parser.add_option('-o', '--original', dest='original_path', default=None,
                      help='Path to original model for comparison (optional)')
    parser.add_option('--original_config', dest='original_config', default=None,
                      help='Path to original model config (optional)')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=False, help='Use CUDA')
    parser.add_option('-H', '--height', dest='height', type='int', default=572,
                      help='Input height for FLOPs calculation')
    parser.add_option('-W', '--width', dest='width', type='int', default=572,
                      help='Input width for FLOPs calculation')
    
    (options, args) = parser.parse_args()
    return options


if __name__ == '__main__':
    args = get_args()
    
    if not args.model_path:
        print("Error: Please provide a model path with -m/--model")
        exit(1)
    
    input_shape = (1, 3, args.height, args.width)
    
    if args.original_path:
        # Compare mode
        compare_models(
            args.original_path, 
            args.model_path,
            args.original_config,
            args.config_path,
            input_shape,
            args.gpu
        )
    else:
        # Single model mode
        print_statistics(args.model_path, args.config_path, input_shape, args.gpu)