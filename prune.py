# prune.py — Carvana (real data) + milesial/Pytorch-UNet pretrained + Prune_U-Net pruning
#
# RELIABILITY FIX (your NVML/CUDACachingAllocator assert):
# - Baseline evaluation is done BEFORE Taylor/Pruner hooks.
#   Your crash happened because you were running eval AFTER a long Taylor loop with
#   many CUDA allocations/hook activity -> allocator/fragmentation issues on some setups.
# - Also: we explicitly delete the Pruner and empty CUDA cache before rebuilding/finetuning.
#
# Output improvements preserved:
# - Pretty per-conv FLOPs before/after pruning
# - Final baseline vs pruned+finetuned summary table (Dice/Params/Size/FLOPs + ratios/factors)
# - Per-part FLOPs and Params comparisons

import os
import os.path as osp
import io
import json
import random
from optparse import OptionParser

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim

from prune_utils import Pruner
from finetune import finetune
from eval import eval_net
from unet import UNet
from utils import (
    get_ids,
    split_ids,
    split_train_val,
    get_imgs_and_masks,
    batch,
    get_logger,
    get_save_dir,
)

# ----------------------------
# Repro helpers
# ----------------------------
def seed_all(seed: int, use_cuda: bool):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


# ----------------------------
# Metrics helpers
# ----------------------------
def count_params(net: nn.Module) -> int:
    return int(sum(p.numel() for p in net.parameters()))

def size_mb_from_state_dict(sd: dict) -> float:
    buf = io.BytesIO()
    torch.save(sd, buf)
    return float(buf.getbuffer().nbytes) / (1024.0 * 1024.0)

def fmt_int(n: int) -> str:
    return f"{int(n):,}"

def pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"

def fmt_mflops(n: int) -> str:
    return f"{(float(n)/1e6):.3f} MFLOPs  ({fmt_int(n)})"

def fmt_gflops(n: int) -> str:
    return f"{(float(n)/1e9):.3f} GFLOPs  ({fmt_int(n)})"

def _ratio(pv, bv):
    if bv == 0:
        return float("nan")
    return pv / bv

def _factor(bv, pv):
    if pv == 0:
        return float("inf")
    return bv / pv


# ----------------------------
# Part grouping
# ----------------------------
def _part_from_name(conv_name: str) -> str:
    if conv_name.startswith("inc."):
        return "inc"
    if conv_name.startswith("down1."):
        return "down1"
    if conv_name.startswith("down2."):
        return "down2"
    if conv_name.startswith("down3."):
        return "down3"
    if conv_name.startswith("down4."):
        return "down4"
    if conv_name.startswith("up1."):
        return "up1"
    if conv_name.startswith("up2."):
        return "up2"
    if conv_name.startswith("up3."):
        return "up3"
    if conv_name.startswith("up4."):
        return "up4"
    if conv_name.startswith("outc."):
        return "outc"
    return "other"

def params_by_part(net: nn.Module) -> dict:
    parts = {k: 0 for k in ["inc","down1","down2","down3","down4","up1","up2","up3","up4","outc","other"]}
    for n, p in net.named_parameters():
        prefix = n.split(".", 1)[0] if "." in n else n
        if prefix in parts:
            parts[prefix] += int(p.numel())
        else:
            parts["other"] += int(p.numel())
    return parts

def summarize_flops_by_part(flops_named: list) -> dict:
    parts = {k: 0 for k in ["inc","down1","down2","down3","down4","up1","up2","up3","up4","outc","other"]}
    for name, fl in flops_named:
        parts[_part_from_name(name)] += int(fl)
    return parts


# ----------------------------
# FLOPs after pruning (no gradients): hook-based, SAME convention as prune_utils.calc_flops
# Convention: H*W*KH*KW*Cin  (NO *Cout)
# ----------------------------
@torch.no_grad()
def conv_flops_named_no_grad(net: nn.Module, x: torch.Tensor) -> list:
    net.eval()
    convs = []
    names = []
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            convs.append(m)
            names.append(name)

    out_hw = {}

    def _hook(name_):
        def fn(module, inputs, output):
            if isinstance(output, torch.Tensor) and output.ndim == 4:
                out_hw[name_] = (int(output.shape[2]), int(output.shape[3]))
        return fn

    hooks = []
    for name, m in zip(names, convs):
        hooks.append(m.register_forward_hook(_hook(name)))

    _ = net(x)

    for h in hooks:
        h.remove()

    report = []
    for name, m in zip(names, convs):
        if name not in out_hw:
            continue
        H, W = out_hw[name]
        _, Cin, KH, KW = m.weight.shape
        fl = int(H * W * KH * KW * Cin)
        report.append((name, fl))
    return report


# ----------------------------
# Printing helpers
# ----------------------------
def _print_flops_per_conv(log, title: str, flops_named: list):
    log.info(title)
    total = 0
    for name, fl in flops_named:
        log.info(f"  - {name:40s}: {fmt_mflops(fl)}")
        total += int(fl)
    log.info(f"  => SUM: {fmt_gflops(total)}")
    return total

def _print_final_summary(log, baseline: dict, pruned: dict):
    log.info("=" * 92)
    log.info("FINAL SUMMARY (Baseline vs Pruned+Finetuned)")
    log.info("=" * 92)
    log.info(f"{'Metric':12s} | {'Baseline':18s} | {'Pruned':18s} | {'Delta':22s} | {'Pruned/Baseline':16s}")
    log.info("-" * 92)

    rows = [
        ("Dice",
         f"{baseline['dice']:.6f}",
         f"{pruned['dice']:.6f}",
         f"{(pruned['dice']-baseline['dice']):+.6f}",
         pct(_ratio(pruned['dice'], baseline['dice']))),

        ("Params",
         fmt_int(baseline['params']),
         fmt_int(pruned['params']),
         f"{pct(_ratio(pruned['params']-baseline['params'], baseline['params']))} ({(pruned['params']-baseline['params']):+d})",
         pct(_ratio(pruned['params'], baseline['params']))),

        ("Size (MB)",
         f"{baseline['size_mb']:.3f}",
         f"{pruned['size_mb']:.3f}",
         f"{pct(_ratio(pruned['size_mb']-baseline['size_mb'], baseline['size_mb']))} ({(pruned['size_mb']-baseline['size_mb']):+.3f})",
         pct(_ratio(pruned['size_mb'], baseline['size_mb']))),

        ("FLOPs",
         f"{baseline['flops_g']:.3f}G",
         f"{pruned['flops_g']:.3f}G",
         f"{pct(_ratio(pruned['flops_g']-baseline['flops_g'], baseline['flops_g']))} ({(pruned['flops_g']-baseline['flops_g']):+.3f}G)",
         pct(_ratio(pruned['flops_g'], baseline['flops_g']))),
    ]

    for r in rows:
        log.info(f"{r[0]:12s} | {r[1]:18s} | {r[2]:18s} | {r[3]:22s} | {r[4]:16s}")

    log.info("-" * 92)
    log.info("Compression factors (Baseline / Pruned):")
    log.info(f"  - Params: {_factor(baseline['params'], pruned['params']):.2f}× smaller")
    log.info(f"  - Size  : {_factor(baseline['size_mb'], pruned['size_mb']):.2f}× smaller")
    log.info(f"  - FLOPs : {_factor(baseline['flops_g'], pruned['flops_g']):.2f}× smaller")
    log.info("=" * 92)

def _print_part_compare(log, title: str, base_parts: dict, pr_parts: dict, unit: str):
    keys = ["inc","down1","down2","down3","down4","up1","up2","up3","up4","outc","other"]
    log.info("=" * 92)
    log.info(title)
    log.info("=" * 92)
    log.info(f"{'Part':10s} | {'Baseline':20s} | {'Pruned':20s} | {'Delta':20s} | {'Pruned/B':12s}")
    log.info("-" * 92)

    for k in keys:
        b = float(base_parts.get(k, 0.0))
        p = float(pr_parts.get(k, 0.0))
        ratio = (p / b) if b > 0 else float("nan")
        delta = p - b

        if unit == "flops":
            b_s = f"{(b/1e9):.3f}G"
            p_s = f"{(p/1e9):.3f}G"
            d_s = f"{(delta/1e9):+.3f}G"
        else:
            b_s = fmt_int(int(b))
            p_s = fmt_int(int(p))
            d_s = f"{int(delta):+d}"

        log.info(f"{k:10s} | {b_s:20s} | {p_s:20s} | {d_s:20s} | {pct(ratio):12s}")

    log.info("-" * 92)


# ----------------------------
# State-dict compact check
# ----------------------------
def compare_state_dicts_compact(ckpt_sd: dict, model_sd: dict):
    model_keys = set(model_sd.keys())
    ckpt_keys = set(ckpt_sd.keys())
    common = model_keys & ckpt_keys

    mismatch = 0
    for k in common:
        if tuple(model_sd[k].shape) != tuple(ckpt_sd[k].shape):
            mismatch += 1

    missing = len(model_keys - ckpt_keys)
    unexpected = len(ckpt_keys - model_keys)
    return len(common), mismatch, missing, unexpected


# ----------------------------
# CLI
# ----------------------------
def get_args():
    p = OptionParser()

    p.add_option("-n", "--name", dest="name", default="initial", help="run name")
    p.add_option("-b", "--batch_size", dest="batch_size", default=2, type="int", help="batch size")
    p.add_option("-t", "--taylor_batches", dest="taylor_batches", default=500, type="int",
                 help="mini-batches for Taylor criterion")
    p.add_option("-p", "--prune_channels", dest="prune_channels", default=300, type="int",
                 help="channels to remove")
    p.add_option("-g", "--gpu", action="store_true", dest="gpu", default=False, help="use cuda")
    p.add_option("-l", "--load", dest="load", default="unet/unet_carvana_scale0.5_epoch2.pth",
                 help="checkpoint path")
    p.add_option("-c", "--channel_txt", dest="channel_txt", default="model_channels.txt",
                 help="channel txt")
    p.add_option("-s", "--scale", dest="scale", type="float", default=0.5, help="downscale factor")
    p.add_option("-r", "--lr", dest="lr", type="float", default=0.1, help="lr finetune")
    p.add_option("-i", "--iters", dest="iters", type="int", default=1500, help="finetune iters")
    p.add_option("-e", "--epochs", dest="epochs", type="int", default=None, help="finetune epochs")
    p.add_option("-f", "--flops", dest="flops_reg", type="float", default=0.001, help="flops reg")

    p.add_option("--seed", dest="seed", type="int", default=0, help="seed")
    p.add_option("--taylor_shuffle", dest="taylor_shuffle", type="int", default=0, help="shuffle in Taylor batches")
    p.add_option("--data_root", dest="data_root", default="data", help="folder containing train/ and train_masks/")
    p.add_option("--eval_baseline", dest="eval_baseline", type="int", default=1, help="1 eval baseline, 0 skip")

    # Cached baseline (used when --eval_baseline 0)
    p.add_option("--baseline_dice", dest="baseline_dice", type="float", default=0.989935,
                 help="cached baseline dice used if baseline eval is skipped")

    p.add_option("--quiet_split", dest="quiet_split", type="int", default=1, help="reduce split logging")
    p.add_option("--print_flops", dest="print_flops", type="int", default=1, help="print flops tables")
    p.add_option("--print_table", dest="print_table", type="int", default=1, help="print final summaries")

    p.add_option("--min_keep_ch", dest="min_keep_ch", type="int", default=1,
             help="Minimum out_channels to keep per Conv2d (prevents zero-channel convs).")
    p.add_option("--min_keep_ratio", dest="min_keep_ratio", type="float", default=0.0,
             help="Optional ratio-based min keep. Effective min is max(min_keep_ch, ceil(out_ch*min_keep_ratio)).")


    p.add_option("--enc_ratio", dest="enc_ratio", type="float", default=0.15,
                 help="Encoder pruning ratio")
    p.add_option("--bn_ratio", dest="bn_ratio", type="float", default=0.45,
                 help="Bottleneck pruning ratio")
    p.add_option("--dec_ratio", dest="dec_ratio", type="float", default=0.40,
                 help="Decoder pruning ratio")


    (opt, _) = p.parse_args()
    return opt


# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    args = get_args()

    data_root = osp.expanduser(args.data_root)
    dir_img = osp.join(data_root, "train") + "/"
    dir_mask = osp.join(data_root, "train_masks") + "/"
    splitfile = osp.join("data", "trainval.json")

    if not osp.isdir(dir_img):
        raise FileNotFoundError(f"train/ not found: {dir_img} (use --data_root)")
    if not osp.isdir(dir_mask):
        raise FileNotFoundError(f"train_masks/ not found: {dir_mask} (use --data_root)")

    device = torch.device("cuda" if (args.gpu and torch.cuda.is_available()) else "cpu")
    seed_all(args.seed, use_cuda=(device.type == "cuda"))

    runname = args.name
    save_path = osp.join("save/", runname)
    save_dir = get_save_dir(save_path, runname, training=False)
    log = get_logger(save_dir, runname)

    log.info(
        "Args: {}".format(
            json.dumps(
                {
                    "batch_size": args.batch_size,
                    "channel_txt": args.channel_txt,
                    "data_root": data_root,
                    "epochs": args.epochs,
                    "eval_baseline": args.eval_baseline,
                    "flops_reg": args.flops_reg,
                    "gpu": (device.type == "cuda"),
                    "iters": args.iters,
                    "load": args.load,
                    "lr": args.lr,
                    "print_flops": int(args.print_flops),
                    "print_table": int(args.print_table),
                    "prune_channels": args.prune_channels,
                    "quiet_split": int(args.quiet_split),
                    "scale": args.scale,
                    "seed": args.seed,
                    "splitfile": splitfile,
                    "taylor_batches": args.taylor_batches,
                    "taylor_shuffle": args.taylor_shuffle,
                },
                indent=4,
                sort_keys=True,
            )
        )
    )

    # ----------------------------
    # Dataset split
    # ----------------------------
    if not osp.exists(splitfile):
        ids = split_ids(get_ids(dir_img))
        iddataset = split_train_val(ids, 0.2, splitfile)
        if int(args.quiet_split) != 1:
            log.info(f"New split dataset -> {splitfile}")
    else:
        with open(splitfile) as f:
            iddataset = json.load(f)
        if int(args.quiet_split) != 1:
            log.info(f"Load split dataset -> {splitfile}")

    train_ids = iddataset["train"]
    val_ids = iddataset["val"]

    # ----------------------------
    # Model init + load
    # ----------------------------
    channel_txt_to_use = args.channel_txt if (args.channel_txt and osp.exists(args.channel_txt)) else None
    net = UNet(n_channels=3, n_classes=2, f_channels=channel_txt_to_use).to(device)
    log.info("Built model using {}...".format(channel_txt_to_use if channel_txt_to_use else "default channels"))

    if args.load and osp.exists(args.load):
        ckpt_sd = torch.load(args.load, map_location=device)
        common, mismatch, missing, unexpected = compare_state_dicts_compact(ckpt_sd, net.state_dict())
        log.info(f"Checkpoint key/shape check: common={common}, mismatch={mismatch}, missing={missing}, unexpected={unexpected}")
        net.load_state_dict(ckpt_sd)
        log.info(f"Loaded checkpoint: {args.load}")
    else:
        log.info(f"Checkpoint not found: {args.load} (continuing)")

    # Capture baseline params/size and per-part params BEFORE pruning
    baseline_param_parts = params_by_part(net)
    baseline_params_total = count_params(net)
    baseline_size_mb = size_mb_from_state_dict(net.state_dict())

    # ----------------------------
    # BASELINE evaluation FIRST (reliability)
    # ----------------------------
    if int(args.eval_baseline) == 1:
        log.info("Evaluating BASELINE model...")
        val_gen = get_imgs_and_masks(val_ids, dir_img, dir_mask, args.scale)
        baseline_dice = eval_net(net, val_gen, len(val_ids), (device.type == "cuda"), args.batch_size)
        seed_all(args.seed, use_cuda=(device.type == "cuda"))
        log.info(f"BASELINE Dice: {baseline_dice:.6f}")
    else:
        baseline_dice = float(args.baseline_dice)
        log.info(f"Skipping baseline eval; using cached baseline dice={baseline_dice:.6f}")

    baseline_metrics = {
        "dice": float(baseline_dice),
        "params": int(baseline_params_total),
        "size_mb": float(baseline_size_mb),
    }

    # ----------------------------
    # Taylor criterion
    # ----------------------------
    pruner = Pruner(net, args.flops_reg, min_keep_ch=int(args.min_keep_ch), min_keep_ratio=float(args.min_keep_ratio))
    criterion = nn.CrossEntropyLoss()
    taylor_shuffle = bool(int(args.taylor_shuffle))

    log.info(f"Taylor criterion: {args.taylor_batches} mini-batches (shuffle={taylor_shuffle})")

    baseline_flops_named = None
    baseline_flops_sum = None
    input_hw = None

    train_gen = get_imgs_and_masks(train_ids, dir_img, dir_mask, args.scale)

    with tqdm(total=args.taylor_batches * args.batch_size) as progress_bar:
        for i, b in enumerate(batch(train_gen, args.batch_size, shuffle=taylor_shuffle)):
            net.zero_grad(set_to_none=True)

            imgs_np = np.array([i0[0] for i0 in b], dtype=np.float32)
            masks_np = np.array([i0[1] for i0 in b])

            imgs = torch.from_numpy(imgs_np).to(device=device, dtype=torch.float32)

            true_masks = torch.from_numpy(masks_np).to(device=device)
            if true_masks.ndim == 4 and true_masks.shape[1] == 1:
                true_masks = true_masks[:, 0, :, :]
            true_masks = (true_masks > 0).long() if true_masks.max() > 1 else true_masks.long()

            if input_hw is None:
                input_hw = (int(imgs.shape[2]), int(imgs.shape[3]))

            out = net(imgs)
            loss = criterion(out, true_masks)
            loss.backward()

            if i == 0:
                # Baseline FLOPs from pruner AFTER first real backward (safe)
                flops_list = pruner.calc_flops()               # aligned with pruner.convs[:-1]
                names = pruner.conv_names[:-1]                 # excludes outc.conv (not hooked in repo)
                baseline_flops_named = list(zip(names, flops_list))
                baseline_flops_sum = int(sum(flops_list))
                if int(args.print_flops) == 1:
                    _print_flops_per_conv(log, "FLOPs before pruning (per conv):", baseline_flops_named)

            pruner.compute_rank()
            progress_bar.update(len(b))

            if (i + 1) >= int(args.taylor_batches):
                log.info("Finished Taylor criterion computation")
                break

    if input_hw is None:
        raise RuntimeError("Could not infer input HW (train set empty?)")

    baseline_metrics["flops_g"] = float(baseline_flops_sum) / 1e9

    # ----------------------------
    # Prune & save
    # ----------------------------

    ratios = {
    "encoder": float(args.enc_ratio),
    "bottleneck": float(args.bn_ratio),
    "decoder": float(args.dec_ratio),
    }

    # safety check
    assert abs(sum(ratios.values()) - 1.0) < 1e-6, \
        f"Ratios must sum to 1.0, got {ratios}"
    
    log.info(
        f"Pruning by parts | "
        f"ENC={ratios['encoder']:.2f}, "
        f"BN={ratios['bottleneck']:.2f}, "
        f"DEC={ratios['decoder']:.2f}, "
        f"TOTAL_CH={args.prune_channels}"
    )
    
    # ============================================================
    # CHANGED LINE (was pruner.pruning)
    # ============================================================
    pruner.prune_by_parts(args.prune_channels, ratios)
    
    log.info(f"Pruning complete: removed {args.prune_channels} channels")

    save_pruned_pth = osp.join(save_dir, "Pruned.pth")
    save_pruned_txt = osp.join(save_dir, "pruned_channels.txt")
    torch.save(net.state_dict(), save_pruned_pth)
    pruner.channel_save(save_pruned_txt)
    log.info(f"Saved pruned weights : {save_pruned_pth}")
    log.info(f"Saved pruned channels: {save_pruned_txt}")

    # IMPORTANT: drop pruner hooks + clear CUDA cache BEFORE doing anything else
    del pruner
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ----------------------------
    # Rebuild pruned net and finetune
    # ----------------------------
    net2 = UNet(n_channels=3, n_classes=2, f_channels=save_pruned_txt).to(device)
    net2.load_state_dict(torch.load(save_pruned_pth, map_location=device))
    log.info("Rebuilt pruned model and loaded weights")

    optimizer = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)

    save_finetuned = osp.join(save_dir, "Finetuned.pth")
    finetune(
        net2,
        optimizer,
        criterion,
        train_ids,
        log,
        save_finetuned,
        args.iters,
        args.epochs,
        args.batch_size,
        (device.type == "cuda"),
        args.scale,
    )

    # ----------------------------
    # Evaluate pruned+finetuned
    # ----------------------------
    val_gen2 = get_imgs_and_masks(val_ids, dir_img, dir_mask, args.scale)
    pruned_dice = eval_net(net2, val_gen2, len(val_ids), (device.type == "cuda"), args.batch_size)
    log.info(f"PRUNED+FINETUNED Dice: {pruned_dice:.6f}")

    # ----------------------------
    # FLOPs after pruning (hook-based, no gradients)
    # ----------------------------
    x_dummy = torch.zeros((1, 3, input_hw[0], input_hw[1]), device=device, dtype=torch.float32)
    pruned_flops_named = conv_flops_named_no_grad(net2, x_dummy)

    # Match baseline convention (exclude outc.conv from totals)
    pruned_flops_named_no_outc = [(n, f) for (n, f) in pruned_flops_named if n != "outc.conv"]
    pruned_flops_sum = int(sum(f for _, f in pruned_flops_named_no_outc))

    if int(args.print_flops) == 1:
        _print_flops_per_conv(log, "FLOPs after pruning (per conv):", pruned_flops_named_no_outc)

    # ----------------------------
    # Collect final metrics
    # ----------------------------
    pruned_params_total = count_params(net2)
    pruned_size_mb = size_mb_from_state_dict(net2.state_dict())

    pruned_metrics = {
        "dice": float(pruned_dice),
        "params": int(pruned_params_total),
        "size_mb": float(pruned_size_mb),
        "flops_g": float(pruned_flops_sum) / 1e9,
    }

    # ----------------------------
    # Final summaries
    # ----------------------------
    if int(args.print_table) == 1:
        _print_final_summary(log, baseline_metrics, pruned_metrics)

        base_flops_parts = summarize_flops_by_part(baseline_flops_named)            # baseline excludes outc
        pr_flops_parts = summarize_flops_by_part(pruned_flops_named_no_outc)
        _print_part_compare(log, "PER-PART FLOPs (Baseline vs Pruned+Finetuned)", base_flops_parts, pr_flops_parts, unit="flops")

        pr_param_parts = params_by_part(net2)
        _print_part_compare(log, "PER-PART Params (Baseline vs Pruned+Finetuned)", baseline_param_parts, pr_param_parts, unit="params")

    # ----------------------------
    # Aggregated groups (Encoder / Bottleneck / Decoder)
    # Put this RIGHT AFTER the per-block breakdown prints
    # ----------------------------

    GROUPS = {
        "ENCODER (inc+down1+down2+down3)": ["inc", "down1", "down2", "down3"],
        "BOTTLENECK (down4)":             ["down4"],
        "DECODER (up1+up2+up3+up4+outc)":  ["up1", "up2", "up3", "up4", "outc"],
    }

    def _sum_keys(d: dict, keys: list) -> float:
        return float(sum(d.get(k, 0) for k in keys))

    log.info("=" * 92)
    log.info("AGGREGATED GROUPS (Baseline vs Pruned+Finetuned)")
    log.info("=" * 92)

    # Params: include outc (it exists in params_by_part)
    params_base_by_group  = {g: _sum_keys(baseline_param_parts, ks) for g, ks in GROUPS.items()}
    params_pruned_by_group = {g: _sum_keys(pr_param_parts, ks)       for g, ks in GROUPS.items()}

    # FLOPs: baseline/pruned FLOPs dicts were built from lists that EXCLUDE outc.conv
    # summarize_flops_by_part still has an "outc" key but it will be 0 in both cases.
    flops_base_by_group   = {g: _sum_keys(base_flops_parts, ks) for g, ks in GROUPS.items()}
    flops_pruned_by_group = {g: _sum_keys(pr_flops_parts, ks)   for g, ks in GROUPS.items()}

    for gname, _ in GROUPS.items():
        pb = params_base_by_group[gname]
        pp = params_pruned_by_group[gname]
        fb = flops_base_by_group[gname]
        fp = flops_pruned_by_group[gname]

        p_keep = (pp / pb * 100.0) if pb > 0 else float("nan")
        f_keep = (fp / fb * 100.0) if fb > 0 else float("nan")

        log.info(f"  - {gname}")
        log.info(
            f"      Params: baseline={pb:,.0f}  pruned={pp:,.0f}  "
            f"keep={p_keep:6.2f}%  reduce={(100.0 - p_keep):6.2f}%"
        )
        log.info(
            f"      FLOPs : baseline={fb:,.0f}  pruned={fp:,.0f}  "
            f"keep={f_keep:6.2f}%  reduce={(100.0 - f_keep):6.2f}%"
        )
