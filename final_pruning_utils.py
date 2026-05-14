"""
final_pruning_utils.py

All utilities and pruning logic for Taylor channel pruning + optional "friendly" (hardware-aligned)
constraints and optional benchmarking.

This file is intended to be imported by final_pruning.py.
"""
from __future__ import annotations

import io
import math
import time
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import os
import json


class Pruner:
    def __init__(self, net, flops_reg, min_keep_ch: int = 1, min_keep_ratio: float = 0.0):
        self.net = net.eval()
        self.flops_reg = flops_reg

        self.min_keep_ch = int(min_keep_ch)
        self.min_keep_ratio = float(min_keep_ratio)

        self.clear_rank()
        self.clear_modules()
        self.clear_cache()

        self._register_hooks()

    def clear_rank(self):
        self.ranks = {}
        self.flops = []

    def clear_modules(self):
        self.convs = []
        self.conv_names = []
        self.BNs = []

        # decoder upsamplers (ConvTranspose2d)
        self.deconvs = {}
        # { "up1.up": module, ... }

    def clear_cache(self):
        self.activation_maps = []
        self.gradients = []

    def _register_hooks(self):
        def forward_hook_fn(module, inputs, output):
            self.activation_maps.append(output.detach())

        def backward_hook_fn(module, grad_input, grad_output):
            self.gradients.append(grad_output[0].detach())

        for name, module in self.net.named_modules():
            if isinstance(module, nn.Conv2d):
                if name != "outc.conv":
                    module.register_forward_hook(forward_hook_fn)
                    module.register_full_backward_hook(backward_hook_fn)
                self.convs.append(module)
                self.conv_names.append(name)

            elif isinstance(module, nn.ConvTranspose2d):
                # store decoder upsamplers by name
                # we only expect up1.up, up2.up, up3.up, up4.up
                self.deconvs[name] = module

            elif isinstance(module, nn.BatchNorm2d):
                self.BNs.append(module)

    def compute_rank(self):
        self.gradients.reverse()

        for layer, act in enumerate(self.activation_maps):
            taylor = (act * self.gradients[layer]).mean(dim=(2, 3)).abs().mean(dim=0)  # [C]
            if layer not in self.ranks:
                self.ranks[layer] = taylor
            else:
                self.ranks[layer] = 0.9 * self.ranks[layer] + 0.1 * taylor

        self.clear_cache()

    def _rank_channels(self, prune_channels):
        """
        Returns:
          sorted_channel_layers: 1D LongTensor [prune_channels] on CPU
          sorted_layer_channels: 1D LongTensor [prune_channels] on CPU
    
        Fixes GPU/CPU indexing mismatch by ensuring:
          - total_rank is on CPU before topk
          - indices are on CPU
          - channel_layers/layer_channels are on CPU
        """
        total_rank_list = []
        channel_layers_list = []
        layer_channels_list = []
    
        # Normalize FLOPs weights (Python list of floats)
        s = float(sum(self.flops)) if len(self.flops) > 0 else 1.0
        if s == 0.0:
            s = 1.0
        self.flops[:] = [float(x) / s for x in self.flops]
    
        # Build ranks (keep everything on CPU for stable indexing)
        for layer, ranks in self.ranks.items():
            # ranks: Tensor [C] possibly on GPU
            taylor = ranks.detach()
            if taylor.is_cuda:
                taylor = taylor.cpu()
    
            denom = torch.sqrt(torch.sum(taylor ** 2)) + 1e-12
            taylor = taylor / denom
    
            # FLOPs penalty term (scalar)
            flops_pen = float(self.flops[layer]) * float(self.flops_reg)
            total_rank_list.append(taylor + flops_pen)
    
            c = int(taylor.shape[0])
            channel_layers_list.extend([int(layer)] * c)
            layer_channels_list.extend(range(c))
    
        # Edge case: nothing ranked
        if len(total_rank_list) == 0:
            return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
    
        channel_layers = torch.tensor(channel_layers_list, dtype=torch.long, device="cpu")
        layer_channels = torch.tensor(layer_channels_list, dtype=torch.long, device="cpu")
        total_rank = torch.cat(total_rank_list, dim=0).to(device="cpu")
    
        k = min(int(prune_channels), int(total_rank.numel()))
        _, sorted_indices = torch.topk(total_rank, k, largest=False)
        sorted_indices = sorted_indices.to(device="cpu")
    
        sorted_channel_layers = channel_layers[sorted_indices]
        sorted_layer_channels = layer_channels[sorted_indices]
        return sorted_channel_layers, sorted_layer_channels

    def create_indices(self):
        chans = [(list(range(c.weight.shape[1])), list(range(c.weight.shape[0]))) for c in self.convs]
        inchans, outchans = list(zip(*chans))
        return list(inchans), list(outchans)

    def _name_to_conv_idx(self):
        return {n: i for i, n in enumerate(self.conv_names)}

    @torch.no_grad()
    def _prune_deconv(self, deconv: nn.ConvTranspose2d, in_keep: list, out_keep: list):
        # ConvTranspose2d weight shape: [in_channels, out_channels, kH, kW]
        w = deconv.weight.data
        deconv.weight.data = w[in_keep, ...][:, out_keep, ...]
        if deconv.bias is not None:
            deconv.bias.data = deconv.bias.data[out_keep]

    def _min_keep_for_layer(self, layer_idx: int, outchans: list) -> int:
        # current out channels count (before we remove in this pruning pass)
        cur = len(outchans[layer_idx])
        # ratio-based min (relative to current)
        ratio_min = int(math.ceil(cur * self.min_keep_ratio)) if self.min_keep_ratio > 0 else 0
        return max(self.min_keep_ch, ratio_min, 1)
    
    def pruning(self, prune_channels):
        sorted_channel_layers, sorted_layer_channels = self._rank_channels(prune_channels)
        inchans, outchans = self.create_indices()

        requested = int(prune_channels)
        applied = 0
        skipped_minkeep = 0
        skipped_missing = 0
        
        # ---- your original dependency logic (unchanged) ----
        for i in range(len(sorted_channel_layers)):
            if applied >= requested:
                break
            
            cl = int(sorted_channel_layers[i])
            lc = int(sorted_layer_channels[i])

            # If channel already removed by earlier dependency ops, skip
            if lc not in outchans[cl]:
                skipped_missing +=1
                continue

            # Enforce min-keep on THIS layer
            min_keep = self._min_keep_for_layer(cl, outchans)
            if len(outchans[cl]) <= min_keep:
                skipped_minkeep += 1
                continue


            # ---- your original dependency logic (unchanged) ----
            res = True if cl in [1, 3, 5, 7] else False
            offset = True if cl in [9, 11, 13, 15] else False
        
            if offset:
                mapping = {9: 7, 11: 5, 13: 3, 15: 1}
                top = self.convs[mapping[cl]].weight.shape[0]
                try:
                    inchans[cl + 1].remove(top + lc)
                except ValueError:
                    pass
            else:
                try:
                    inchans[cl + 1].remove(lc)
                except ValueError:
                    pass
        
            if res:
                try:
                    inchans[-(cl + 2)].remove(lc)
                except ValueError:
                    pass
        
            # finally remove from outchans
            try:
                outchans[cl].remove(lc)
                applied += 1
            except ValueError:
                skipped_missing += 1
                continue
        
        # If we couldn't apply full budget, log it (print is fine here; you can use logger outside)
        if applied < requested:
            print(
                f"[Pruner] Requested prune={requested}, applied={applied}. "
                f"Skipped(min_keep)={skipped_minkeep}, skipped(already_removed)={skipped_missing}. "
                f"min_keep_ch={self.min_keep_ch}, min_keep_ratio={self.min_keep_ratio}"
            )
        

        # -----------------------------------------------------------------------------
        # FORCE DECODER CONCAT CONSISTENCY
        # For each Up block, the input to upX.conv.double_conv.0 is:
        #   cat([skip, upsample], dim=1)
        # So inchans(conv0) must be: keep_skip + (half + keep_skip)
        # where "half" is the original skip width (pre-slice) = conv0.in_channels // 2.
        # -----------------------------------------------------------------------------
        name_to_idx = self._name_to_conv_idx()

        def _force_up_conv0_in(up_conv0_name: str, skip_out_name: str):
            idx_conv0 = name_to_idx[up_conv0_name]
            idx_skip  = name_to_idx[skip_out_name]

            keep_skip = outchans[idx_skip]  # indices relative to skip tensor (0..half-1)
            half = self.convs[idx_conv0].weight.shape[1] // 2  # original half-width (pre-slice)

            # concat order in your forward: torch.cat([x2(skip), x1(up)], dim=1)
            inchans[idx_conv0] = keep_skip + [half + k for k in keep_skip]

        _force_up_conv0_in("up1.conv.double_conv.0", "down3.maxpool_conv.1.double_conv.3")
        _force_up_conv0_in("up2.conv.double_conv.0", "down2.maxpool_conv.1.double_conv.3")
        _force_up_conv0_in("up3.conv.double_conv.0", "down1.maxpool_conv.1.double_conv.3")
        _force_up_conv0_in("up4.conv.double_conv.0", "inc.double_conv.3")

        # Safety: ensure no conv got emptied by forced adjustments
        for li in range(len(outchans)):
            if len(outchans[li]) < 1:
                raise RuntimeError(f"[Pruner] Invalid pruning result: layer {li} ({self.conv_names[li]}) has 0 out channels.")


        # ---- prune Conv2d params ----
        for i, c in enumerate(self.convs):
            self.convs[i].weight.data = c.weight.data[outchans[i], ...][:, inchans[i], ...]
            if c.bias is not None:
                self.convs[i].bias.data = c.bias.data[outchans[i]]

        # ---- prune BN params ----
        for i, bn in enumerate(self.BNs):
            self.BNs[i].running_mean.data = bn.running_mean.data[outchans[i]]
            self.BNs[i].running_var.data = bn.running_var.data[outchans[i]]
            self.BNs[i].weight.data = bn.weight.data[outchans[i]]
            self.BNs[i].bias.data = bn.bias.data[outchans[i]]

        # ---- NEW: prune ConvTranspose2d (decoder upsamplers) ----
        name_to_idx = self._name_to_conv_idx()

        # We use the *second conv* of each DoubleConv as the tensor output:
        # *.double_conv.3 is the 2nd Conv2d
        # encoder outputs
        idx_x1 = name_to_idx["inc.double_conv.3"]
        idx_x2 = name_to_idx["down1.maxpool_conv.1.double_conv.3"]
        idx_x3 = name_to_idx["down2.maxpool_conv.1.double_conv.3"]
        idx_x4 = name_to_idx["down3.maxpool_conv.1.double_conv.3"]
        idx_x5 = name_to_idx["down4.maxpool_conv.1.double_conv.3"]

        # decoder block outputs (after conv)
        idx_u1 = name_to_idx["up1.conv.double_conv.3"]
        idx_u2 = name_to_idx["up2.conv.double_conv.3"]
        idx_u3 = name_to_idx["up3.conv.double_conv.3"]
        # up4 output exists too but not used as input to another deconv
        # idx_u4 = name_to_idx["up4.conv.double_conv.3"]

        # deconvs: input comes from deeper tensor, output should match skip tensor
        # up1.up: x5 -> (match x4)
        if "up1.up" in self.deconvs:
            self._prune_deconv(self.deconvs["up1.up"], in_keep=outchans[idx_x5], out_keep=outchans[idx_x4])

        # up2.up: up1_out -> (match x3)
        if "up2.up" in self.deconvs:
            self._prune_deconv(self.deconvs["up2.up"], in_keep=outchans[idx_u1], out_keep=outchans[idx_x3])

        # up3.up: up2_out -> (match x2)
        if "up3.up" in self.deconvs:
            self._prune_deconv(self.deconvs["up3.up"], in_keep=outchans[idx_u2], out_keep=outchans[idx_x2])

        # up4.up: up3_out -> (match x1)
        if "up4.up" in self.deconvs:
            self._prune_deconv(self.deconvs["up4.up"], in_keep=outchans[idx_u3], out_keep=outchans[idx_x1])

    def channel_save(self, path: str):
        n = self.net
        chans = []

        # inc: mid, out
        chans.append(n.inc.double_conv[0].weight.shape[0])  # mid
        chans.append(n.inc.double_conv[3].weight.shape[0])  # out

        # downs: mid, out
        for d in [n.down1, n.down2, n.down3, n.down4]:
            c0 = d.maxpool_conv[1].double_conv[0]  # in->mid
            c1 = d.maxpool_conv[1].double_conv[3]  # mid->out
            chans.append(c0.weight.shape[0])       # mid
            chans.append(c1.weight.shape[0])       # out

        # ups: deep_in, mid, out
        for u in [n.up1, n.up2, n.up3, n.up4]:
            # ConvTranspose2d weight: [deep_in, skip_ch, 2, 2]
            chans.append(u.up.weight.shape[0])                 # deep_in
            chans.append(u.conv.double_conv[0].weight.shape[0])  # mid
            chans.append(u.conv.double_conv[3].weight.shape[0])  # out

        assert len(chans) == 22, f"Expected 22 integers, got {len(chans)}: {chans}"

        with open(path, "w") as f:
            for c in chans:
                f.write(f"{int(c)}\n")




    def calc_flops(self):
        self.flops = []
        for i, c in enumerate(self.convs[:-1]):
            H, W = self.gradients[i].shape[2:]
            _, I, KH, KW = c.weight.shape
            self.flops.append(H * W * KH * KW * I)
        return self.flops


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


def ceil_to_multiple(x: int, m: int) -> int:
    if m <= 0:
        return int(x)
    x = int(x)
    return int(((x + m - 1) // m) * m)


def floor_to_multiple(x: int, m: int) -> int:
    if m <= 0:
        return int(x)
    x = int(x)
    return int((x // m) * m)


# ----------------------------
# Part grouping (printing)
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


def conv_out_channels_by_name(net: nn.Module) -> Dict[str, int]:
    out = {}
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            out[name] = int(m.out_channels)
    return out


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
# Taylor mode helpers (FIX #1)
# ----------------------------
def set_model_for_taylor(net: nn.Module, freeze_bn: bool = True):
    net.train()
    if freeze_bn:
        for m in net.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                m.eval()


# ----------------------------
# Group budget (selection scope)
# ----------------------------
@dataclass

class GroupBudget:
    prune_down_total: Optional[int] = None  # pooled encoder+bottleneck
    prune_encoder: int = 0
    prune_bottleneck: int = 0
    prune_decoder: int = 0

    def total(self) -> int:
        if self.prune_down_total is not None:
            return int(self.prune_down_total) + int(self.prune_decoder)
        return int(self.prune_encoder) + int(self.prune_bottleneck) + int(self.prune_decoder)


def make_group_budget(
    prune_down_total: Optional[int] = None,
    prune_encoder: int = 0,
    prune_bottleneck: int = 0,
    prune_decoder: int = 0,
) -> GroupBudget:
    if prune_down_total is not None:
        return GroupBudget(
            prune_down_total=int(prune_down_total),
            prune_encoder=0,
            prune_bottleneck=0,
            prune_decoder=int(prune_decoder),
        )
    return GroupBudget(
        prune_down_total=None,
        prune_encoder=int(prune_encoder),
        prune_bottleneck=int(prune_bottleneck),
        prune_decoder=int(prune_decoder),
    )


# ----------------------------
# SAFE pruner base (same as fixed_pruning.py logic)
# ----------------------------


def resolve_budget_and_requested(scope: str,
                                prune_channels: int,
                                prune_encoder: int = 0,
                                prune_bottleneck: int = 0,
                                prune_decoder: int = 0,
                                prune_down_total: Optional[int] = None
                                ) -> Tuple[Optional["GroupBudget"], int, str]:
    """Resolve the pruning budget given a scope and user budgets.

    Returns:
        (budget, requested_total, scope_desc)

    Scopes:
      - global: uses prune_channels
      - encoder|bottleneck|decoder: uses prune_channels for that single part
      - custom: uses (prune_encoder, prune_bottleneck, prune_decoder)
      - pooled_down: uses prune_down_total + prune_decoder
    """
    sc = str(scope).lower().strip()
    if sc == "global":
        return None, int(prune_channels), "GLOBAL"

    if sc in ("encoder", "bottleneck", "decoder"):
        b = make_group_budget(
            prune_encoder=int(prune_channels) if sc == "encoder" else 0,
            prune_bottleneck=int(prune_channels) if sc == "bottleneck" else 0,
            prune_decoder=int(prune_channels) if sc == "decoder" else 0,
        )
        return b, int(b.total()), sc.upper()

    if sc == "custom":
        pe = int(prune_encoder); pb = int(prune_bottleneck); pd = int(prune_decoder)
        if pe + pb + pd <= 0:
            raise ValueError("custom scope requires at least one of --prune_encoder/--prune_bottleneck/--prune_decoder > 0")
        b = make_group_budget(prune_encoder=pe, prune_bottleneck=pb, prune_decoder=pd)
        return b, int(b.total()), "CUSTOM (encoder/bottleneck/decoder)"

    if sc == "pooled_down":
        if prune_down_total is None:
            raise ValueError("pooled_down scope requires --prune_down_total")
        d = int(prune_down_total)
        if d < 0:
            raise ValueError("pooled_down scope requires --prune_down_total >= 0")
        pd = int(prune_decoder)
        b = make_group_budget(prune_down_total=d, prune_decoder=pd)
        return b, int(b.total()), "POOLED-DOWN (encoder+bottleneck) + decoder"

    raise ValueError(f"Unknown scope: {scope}")


class SafeBudgetPruner(Pruner):
    """
    Same logic as fixed_pruning.py:
      - abs/square ranks to avoid negative catastrophe
      - group scopes and budget scopes
      - debug printing: rank stats + selection distribution
    """

    def __init__(
        self,
        net,
        flops_reg: float,
        scope: str = "global",
        include_inc: bool = True,
        budget: Optional[GroupBudget] = None,
        rank_transform: str = "abs",     # "abs" | "square" | "none"
        rank_overfetch: int = 5,
        max_candidates_per_layer: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(net, flops_reg, **kwargs)
        self.scope = str(scope)
        self.include_inc = bool(include_inc)
        self.budget = budget
        self.rank_transform = str(rank_transform)
        self.rank_overfetch = max(1, int(rank_overfetch))
        self.max_candidates_per_layer = max_candidates_per_layer
        self._last_selected: List[Tuple[float, int, int]] = []

    # ----- group membership from conv name -----
    def _group_of_layer(self, layer_idx: int) -> str:
        name = self.conv_names[layer_idx] if layer_idx < len(self.conv_names) else ""
        if name.startswith("outc."):
            return "other"
        if name.startswith("inc."):
            return "encoder" if self.include_inc else "other"
        if name.startswith("down1.") or name.startswith("down2.") or name.startswith("down3."):
            return "encoder"
        if name.startswith("down4."):
            return "bottleneck"
        if name.startswith("up1.") or name.startswith("up2.") or name.startswith("up3.") or name.startswith("up4."):
            return "decoder"
        return "other"

    def _is_down_layer(self, layer_idx: int) -> bool:
        return self._group_of_layer(layer_idx) in ("encoder", "bottleneck")

    def _apply_rank_transform(self, t: torch.Tensor) -> torch.Tensor:
        if self.rank_transform == "abs":
            return t.abs()
        if self.rank_transform == "square":
            return t.pow(2)
        return t

    def _norm_flops(self) -> List[float]:
        if len(self.flops) == 0:
            return []
        s = float(sum(self.flops))
        if s == 0.0:
            s = 1.0
        return [float(x) / s for x in self.flops]

    def _build_rank_items(self) -> List[Tuple[float, int, int]]:
        items: List[Tuple[float, int, int]] = []
        flops_norm = self._norm_flops()

        for layer_idx, ranks in self.ranks.items():
            name = self.conv_names[layer_idx] if layer_idx < len(self.conv_names) else ""
            if name.startswith("outc."):
                continue

            taylor = ranks.detach()
            if taylor.is_cuda:
                taylor = taylor.cpu()

            taylor = self._apply_rank_transform(taylor)

            denom = torch.sqrt(torch.sum(taylor ** 2)) + 1e-12
            taylor = taylor / denom

            flops_pen = 0.0
            if layer_idx < len(flops_norm):
                flops_pen = float(flops_norm[layer_idx]) * float(self.flops_reg)

            for ch in range(int(taylor.numel())):
                score = float(taylor[ch].item()) + float(flops_pen)
                items.append((score, int(layer_idx), int(ch)))

        return items

    @staticmethod
    def _select_k_lowest(items: List[Tuple[float, int, int]], k: int) -> List[Tuple[float, int, int]]:
        if k <= 0 or len(items) == 0:
            return []
        k = min(int(k), len(items))
        items_sorted = sorted(items, key=lambda x: x[0])
        return items_sorted[:k]

    def _filter_items_by_scope(self, items: List[Tuple[float, int, int]]) -> List[Tuple[float, int, int]]:
        if self.scope == "global":
            return items

        allowed_groups = {"encoder", "bottleneck", "decoder"}
        kept = [(s, l, c) for (s, l, c) in items if self._group_of_layer(l) in allowed_groups]

        if self.scope == "encoder":
            kept = [(s, l, c) for (s, l, c) in kept if self._group_of_layer(l) == "encoder"]
        elif self.scope == "bottleneck":
            kept = [(s, l, c) for (s, l, c) in kept if self._group_of_layer(l) == "bottleneck"]
        elif self.scope == "decoder":
            kept = [(s, l, c) for (s, l, c) in kept if self._group_of_layer(l) == "decoder"]
        elif self.scope in ("custom", "pooled_down"):
            pass

        return kept

    def log_rank_stats(self, log, max_layers: int = 200):
        log.info("=" * 92)
        log.info("RANK STATS (per layer) — helps detect catastrophic selection causes")
        log.info("=" * 92)

        layer_idxs = sorted(self.ranks.keys())
        if len(layer_idxs) > max_layers:
            layer_idxs = layer_idxs[:max_layers]

        flops_norm = self._norm_flops()

        for li in layer_idxs:
            name = self.conv_names[li] if li < len(self.conv_names) else f"layer{li}"
            if name.startswith("outc."):
                continue
            conv = self.convs[li] if li < len(self.convs) else None
            out_ch = int(conv.out_channels) if conv is not None else -1
            g = self._group_of_layer(li)

            r = self.ranks[li].detach()
            r_cpu = r.cpu() if r.is_cuda else r
            neg_frac = float((r_cpu < 0).float().mean().item()) if r_cpu.numel() > 0 else 0.0

            rt = self._apply_rank_transform(r_cpu)
            mn = float(rt.min().item()) if rt.numel() > 0 else 0.0
            md = float(rt.median().item()) if rt.numel() > 0 else 0.0
            mx = float(rt.max().item()) if rt.numel() > 0 else 0.0
            fp = float(flops_norm[li]) if li < len(flops_norm) else 0.0

            log.info(f"  - {name:40s} | group={g:10s} | out_ch={out_ch:4d} | "
                     f"neg%={neg_frac*100:6.2f} | rank(min/med/max)={mn:.3e}/{md:.3e}/{mx:.3e} | flops_w={fp:.4f}")

        log.info("=" * 92)

    def log_selection_distribution(self, log, selected: List[Tuple[float, int, int]]):
        by_conv: Dict[str, int] = {}
        by_group: Dict[str, int] = {"encoder": 0, "bottleneck": 0, "decoder": 0, "other": 0}

        for _, li, _ in selected:
            name = self.conv_names[li] if li < len(self.conv_names) else f"layer{li}"
            by_conv[name] = by_conv.get(name, 0) + 1
            g = self._group_of_layer(li)
            by_group[g] = by_group.get(g, 0) + 1

        log.info("=" * 92)
        log.info("SELECTION DISTRIBUTION (how many channels picked per conv / group)")
        log.info("=" * 92)
        log.info(f"Group counts: {json.dumps(by_group, sort_keys=True)}")

        top = sorted(by_conv.items(), key=lambda x: x[1], reverse=True)[:25]
        log.info("Top convs by selected channels:")
        for n, c in top:
            log.info(f"  - {n:40s}: {c}")
        log.info("=" * 92)

    def preview_selection(self, requested_prune: int) -> Tuple[torch.Tensor, torch.Tensor]:
        a, b = self._rank_channels(requested_prune)
        return a, b

    def _rank_channels(self, prune_channels: int):
        items_all = self._build_rank_items()
        items = self._filter_items_by_scope(items_all)

        if len(items) == 0:
            self._last_selected = []
            return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)

        of = int(self.rank_overfetch)

        if self.scope in ("custom", "pooled_down"):
            if self.budget is None:
                raise ValueError("Budget scope selected but budget is None.")

            budget = self.budget
            chosen: List[Tuple[float, int, int]] = []

            def _k(req: int, pool_len: int) -> int:
                if req <= 0 or pool_len <= 0:
                    return 0
                return min(pool_len, int(req) * of)

            if budget.prune_down_total is not None:
                down_items = [(s, l, c) for (s, l, c) in items if self._is_down_layer(l)]
                dec_items = [(s, l, c) for (s, l, c) in items if self._group_of_layer(l) == "decoder"]
                chosen += self._select_k_lowest(down_items, _k(int(budget.prune_down_total), len(down_items)))
                chosen += self._select_k_lowest(dec_items, _k(int(budget.prune_decoder), len(dec_items)))
            else:
                enc_items = [(s, l, c) for (s, l, c) in items if self._group_of_layer(l) == "encoder"]
                bot_items = [(s, l, c) for (s, l, c) in items if self._group_of_layer(l) == "bottleneck"]
                dec_items = [(s, l, c) for (s, l, c) in items if self._group_of_layer(l) == "decoder"]
                chosen += self._select_k_lowest(enc_items, _k(int(budget.prune_encoder), len(enc_items)))
                chosen += self._select_k_lowest(bot_items, _k(int(budget.prune_bottleneck), len(bot_items)))
                chosen += self._select_k_lowest(dec_items, _k(int(budget.prune_decoder), len(dec_items)))

            if len(chosen) == 0:
                self._last_selected = []
                return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)

            chosen = sorted(chosen, key=lambda x: x[0])

        else:
            k = int(prune_channels) * of
            items_sorted = sorted(items, key=lambda x: x[0])

            if self.max_candidates_per_layer is None:
                chosen = items_sorted[: min(k, len(items_sorted))]
            else:
                cap = int(self.max_candidates_per_layer)
                per_layer = {}
                chosen = []
                for it in items_sorted:
                    _, li, _ = it
                    per_layer[li] = per_layer.get(li, 0)
                    if per_layer[li] >= cap:
                        continue
                    chosen.append(it)
                    per_layer[li] += 1
                    if len(chosen) >= min(k, len(items_sorted)):
                        break

        seen = set()
        dedup: List[Tuple[float, int, int]] = []
        for s, l, c in chosen:
            key = (int(l), int(c))
            if key in seen:
                continue
            seen.add(key)
            dedup.append((float(s), int(l), int(c)))

        self._last_selected = dedup
        layer_tensor = torch.tensor([l for (_, l, _) in dedup], dtype=torch.long, device="cpu")
        chan_tensor = torch.tensor([c for (_, _, c) in dedup], dtype=torch.long, device="cpu")
        return layer_tensor, chan_tensor


# ----------------------------
# FRIENDLY pruner: prune in blocks of M per layer
# ----------------------------

class FriendlyBudgetPruner(SafeBudgetPruner):
    """
    Adds "friendly" selection:
      - select prune candidates in blocks of `friendly_multiple` channels per conv
      - ensures each conv's pruned count is a multiple of M (baseline widths are typically multiples)
      - this yields final out_channels that remain multiples of M.
    """

    def __init__(
        self,
        net,
        flops_reg: float,
        friendly_enabled: bool = True,
        friendly_multiple: int = 8,
        friendly_round: str = "up",   # "up" to prune at least requested; "down" to prune at most requested
        **kwargs,
    ):
        super().__init__(net, flops_reg, **kwargs)
        self.friendly_enabled = bool(friendly_enabled)
        self.friendly_multiple = int(max(1, friendly_multiple))
        self.friendly_round = str(friendly_round).lower().strip()
        self._effective_prune_last: Optional[int] = None

    def _layer_sorted_candidates(self) -> Dict[int, List[Tuple[float, int]]]:
        """
        Returns dict[layer_idx] -> list of (score, ch_idx) sorted ascending (low score pruned first).
        Score definition matches SafeBudgetPruner: normalized(taylor) + flops_pen.
        """
        flops_norm = self._norm_flops()
        out: Dict[int, List[Tuple[float, int]]] = {}

        for layer_idx, ranks in self.ranks.items():
            name = self.conv_names[layer_idx] if layer_idx < len(self.conv_names) else ""
            if name.startswith("outc."):
                continue

            taylor = ranks.detach()
            if taylor.is_cuda:
                taylor = taylor.cpu()

            taylor = self._apply_rank_transform(taylor)

            denom = torch.sqrt(torch.sum(taylor ** 2)) + 1e-12
            taylor = taylor / denom

            flops_pen = 0.0
            if layer_idx < len(flops_norm):
                flops_pen = float(flops_norm[layer_idx]) * float(self.flops_reg)

            scores = (taylor + float(flops_pen)).tolist()
            pairs = [(float(scores[i]), int(i)) for i in range(len(scores))]
            pairs.sort(key=lambda x: x[0])
            out[int(layer_idx)] = pairs

        return out

    def _eligible_layers_for_scope(self, layer_candidates: Dict[int, List[Tuple[float, int]]]) -> List[int]:
        """
        Filter layers by scope (global/encoder/bottleneck/decoder/custom/pooled_down).
        For custom/pooled_down, layer filtering is handled per-group in _rank_channels.
        """
        if self.scope == "global":
            return sorted(layer_candidates.keys())

        allowed_groups = {"encoder", "bottleneck", "decoder"}
        layers = [li for li in layer_candidates.keys() if self._group_of_layer(li) in allowed_groups]

        if self.scope == "encoder":
            layers = [li for li in layers if self._group_of_layer(li) == "encoder"]
        elif self.scope == "bottleneck":
            layers = [li for li in layers if self._group_of_layer(li) == "bottleneck"]
        elif self.scope == "decoder":
            layers = [li for li in layers if self._group_of_layer(li) == "decoder"]
        elif self.scope in ("custom", "pooled_down"):
            pass

        return sorted(layers)

    def _layer_max_prune(self, layer_idx: int) -> int:
        """
        Computes max prunable channels for a layer under (min_keep_ch, min_keep_ratio) + friendly blocks.
        """
        conv = self.convs[layer_idx] if layer_idx < len(self.convs) else None
        if conv is None:
            return 0

        out_ch = int(conv.out_channels)
        # outc.* is excluded already; but keep robust:
        name = self.conv_names[layer_idx] if layer_idx < len(self.conv_names) else ""
        if name.startswith("outc."):
            return 0

        min_keep_ratio = float(getattr(self, "min_keep_ratio", 0.0))
        min_keep_ch = int(getattr(self, "min_keep_ch", 0))

        min_keep_by_ratio = int(math.ceil(min_keep_ratio * out_ch)) if min_keep_ratio > 0 else 0
        min_keep = max(int(min_keep_ch), int(min_keep_by_ratio))
        min_keep = min(min_keep, out_ch)  # avoid negative
        max_prune = max(0, out_ch - min_keep)

        if self.friendly_enabled and self.friendly_multiple > 1 and out_ch >= self.friendly_multiple:
            # prune in whole blocks of M (keeps out_ch multiple if it started multiple)
            max_prune = floor_to_multiple(max_prune, self.friendly_multiple)

        return int(max_prune)

    @staticmethod
    def _select_blocks_greedy(
        layer_candidates: Dict[int, List[Tuple[float, int]]],
        layer_max_prune: Dict[int, int],
        target_prune: int,
        block: int,
    ) -> List[Tuple[float, int, int]]:
        """
        Greedy block selection across layers:
          - Each step picks the next best block (sum of next `block` scores) from any layer.
          - Returns list of (score, layer_idx, ch_idx) flattened into per-channel entries.
        """
        if target_prune <= 0:
            return []
        if block <= 0:
            return []

        # Build per-layer prefix sums for the allowed range
        per_layer = {}
        for li, pairs in layer_candidates.items():
            cap = int(layer_max_prune.get(li, 0))
            if cap <= 0:
                continue
            cap = min(cap, len(pairs))
            if cap <= 0:
                continue

            # take lowest `cap` channels
            use = pairs[:cap]
            scores = [p[0] for p in use]
            chans  = [p[1] for p in use]

            # only full blocks count
            n_full = cap // block
            if n_full <= 0:
                continue

            # prefix sum for block cost
            ps = [0.0]
            for s in scores[: n_full * block]:
                ps.append(ps[-1] + float(s))

            per_layer[li] = {
                "scores": scores,
                "chans": chans,
                "n_full_blocks": n_full,
                "ps": ps,
                "bidx": 0,
            }

        if not per_layer:
            return []

        # Min-heap of (block_cost, layer_idx)
        import heapq
        heap = []

        def block_cost(li: int, bidx: int) -> float:
            st = bidx * block
            en = (bidx + 1) * block
            ps = per_layer[li]["ps"]
            return float(ps[en] - ps[st])

        for li, d in per_layer.items():
            heapq.heappush(heap, (block_cost(li, 0), li))

        selected: List[Tuple[float, int, int]] = []
        need = int(target_prune)

        while heap and len(selected) < need:
            _, li = heapq.heappop(heap)
            d = per_layer[li]
            bidx = int(d["bidx"])
            if bidx >= int(d["n_full_blocks"]):
                continue

            st = bidx * block
            en = (bidx + 1) * block

            # add the whole block channels
            for j in range(st, en):
                # store per-channel with its individual score (for debug distribution)
                selected.append((float(d["scores"][j]), int(li), int(d["chans"][j])))
                if len(selected) >= need:
                    break

            d["bidx"] = bidx + 1

            # push next block for same layer
            if d["bidx"] < int(d["n_full_blocks"]) and len(selected) < need:
                heapq.heappush(heap, (block_cost(li, d["bidx"]), li))

        return selected

    def _effective_target(self, requested: int) -> int:
        if not self.friendly_enabled or self.friendly_multiple <= 1:
            return int(requested)

        if self.friendly_round == "down":
            return floor_to_multiple(int(requested), int(self.friendly_multiple))
        # default: round UP to prune at least requested
        return ceil_to_multiple(int(requested), int(self.friendly_multiple))

    def _rank_channels(self, prune_channels: int):
        """
        Friendly selection:
          - determine target prune (rounded to multiple M if enabled)
          - for each scope/budget, select in blocks of M per layer (greedy by block cost)
        """
        # fallback to original behavior if disabled
        if not self.friendly_enabled or self.friendly_multiple <= 1:
            return super()._rank_channels(prune_channels)

        M = int(self.friendly_multiple)
        requested = int(prune_channels)
        target = self._effective_target(requested)
        self._effective_prune_last = int(target)

        layer_candidates_all = self._layer_sorted_candidates()

        def build_for_layers(layers: List[int], tgt: int) -> List[Tuple[float, int, int]]:
            lc = {li: layer_candidates_all[li] for li in layers if li in layer_candidates_all}
            lmax = {li: self._layer_max_prune(li) for li in layers}
            # Ensure target does not exceed total capacity (rounded down to multiple)
            cap_total = int(sum(lmax.values()))
            if cap_total <= 0:
                return []
            # Keep target within capacity, preserving block structure
            tgt_eff = min(int(tgt), int(cap_total))
            tgt_eff = floor_to_multiple(tgt_eff, M)
            return self._select_blocks_greedy(lc, lmax, tgt_eff, M)

        # ------------------------
        # Budget scopes
        # ------------------------
        chosen: List[Tuple[float, int, int]] = []

        if self.scope in ("custom", "pooled_down"):
            if self.budget is None:
                raise ValueError("Budget scope selected but budget is None.")

            b = self.budget

            if b.prune_down_total is not None:
                down_layers = [li for li in layer_candidates_all.keys() if self._is_down_layer(li)]
                dec_layers  = [li for li in layer_candidates_all.keys() if self._group_of_layer(li) == "decoder"]

                tgt_down = self._effective_target(int(b.prune_down_total))
                tgt_dec  = self._effective_target(int(b.prune_decoder))

                chosen += build_for_layers(sorted(down_layers), tgt_down)
                chosen += build_for_layers(sorted(dec_layers),  tgt_dec)
            else:
                enc_layers = [li for li in layer_candidates_all.keys() if self._group_of_layer(li) == "encoder"]
                bot_layers = [li for li in layer_candidates_all.keys() if self._group_of_layer(li) == "bottleneck"]
                dec_layers = [li for li in layer_candidates_all.keys() if self._group_of_layer(li) == "decoder"]

                tgt_enc = self._effective_target(int(b.prune_encoder))
                tgt_bot = self._effective_target(int(b.prune_bottleneck))
                tgt_dec = self._effective_target(int(b.prune_decoder))

                chosen += build_for_layers(sorted(enc_layers), tgt_enc)
                chosen += build_for_layers(sorted(bot_layers), tgt_bot)
                chosen += build_for_layers(sorted(dec_layers), tgt_dec)

            # If user asked global "prune_channels", ignore here; budget controls amounts.
            # (Consistent with fixed_pruning.py behavior.)

        else:
            # ------------------------
            # Global / single-part scopes
            # ------------------------
            layers = self._eligible_layers_for_scope(layer_candidates_all)
            chosen = build_for_layers(layers, target)

        # Dedup (layer, ch)
        seen = set()
        dedup: List[Tuple[float, int, int]] = []
        for s, l, c in chosen:
            key = (int(l), int(c))
            if key in seen:
                continue
            seen.add(key)
            dedup.append((float(s), int(l), int(c)))

        # If we rounded UP, we want to prune exactly that target. If we rounded DOWN by capacity,
        # dedup length may be < target; that's acceptable and will be reflected in logs.
        self._last_selected = dedup

        layer_tensor = torch.tensor([l for (_, l, _) in dedup], dtype=torch.long, device="cpu")
        chan_tensor  = torch.tensor([c for (_, _, c) in dedup], dtype=torch.long, device="cpu")
        return layer_tensor, chan_tensor



# ----------------------------
# Custom sequential (total + pct) helpers
# ----------------------------
@dataclass
class DepSnapshot:
    """Minimal channel-dimension snapshot to measure dependency-induced pruning."""
    down4_in: int  # bottleneck first conv input channels
    dec_deconv_in: int  # sum of decoder ConvTranspose2d in_channels
    dec_deconv_out: int  # sum of decoder ConvTranspose2d out_channels


def compute_targets_from_pct(total: int, enc_pct: float, bot_pct: float, dec_pct: float) -> Tuple[int, int, int]:
    """Split a TOTAL channel budget into (enc, bottleneck, dec) integer targets.

    Accepts percentages (e.g., 20/60/20) or fractions (e.g., 0.2/0.6/0.2).
    Ensures enc+bot+dec == total by assigning remainder to decoder.
    """
    total = int(total)
    if total < 0:
        raise ValueError("total budget must be >= 0")

    p = [float(enc_pct), float(bot_pct), float(dec_pct)]
    # interpret as fractions if they look like fractions
    if all(0.0 <= x <= 1.0 for x in p) and sum(p) <= 1.01:
        p = [x * 100.0 for x in p]

    s = float(sum(p))
    if s <= 0:
        return 0, 0, 0

    # normalize if not exactly 100
    p = [x / s for x in p]

    enc = int(total * p[0])
    bot = int(total * p[1])
    dec = int(total - enc - bot)
    return max(enc, 0), max(bot, 0), max(dec, 0)


def take_dep_snapshot(net: nn.Module) -> DepSnapshot:
    """Snapshot channel dimensions that change due to U-Net dependencies.

    - down4_in captures encoder->bottleneck dependency (down3 out prunes down4 in).
    - decoder deconvs capture both skip-width effects (out_channels) and upsample-path effects (in_channels).
    """
    down4_in = int(net.down4.maxpool_conv[1].double_conv[0].weight.shape[1])

    dec_in = 0
    dec_out = 0
    for i in (1, 2, 3, 4):
        up = getattr(net, f"up{i}")
        # ConvTranspose2d weight shape: [in_channels, out_channels, kH, kW]
        dec_in += int(up.up.weight.shape[0])
        dec_out += int(up.up.weight.shape[1])

    return DepSnapshot(down4_in=down4_in, dec_deconv_in=dec_in, dec_deconv_out=dec_out)


def dep_pruned_counts(base: DepSnapshot, cur: DepSnapshot) -> dict:
    """Compute dependency-induced channel reductions vs baseline snapshot."""
    bot_dep = max(0, int(base.down4_in) - int(cur.down4_in))
    dec_dep = max(
        0,
        (int(base.dec_deconv_in) - int(cur.dec_deconv_in))
        + (int(base.dec_deconv_out) - int(cur.dec_deconv_out)),
    )
    return {"bottleneck_dep": int(bot_dep), "decoder_dep": int(dec_dep)}



# ----------------------------
# Inference benchmarking
# ----------------------------
@torch.no_grad()

def benchmark_inference_avg_ms(
    net: nn.Module,
    device: torch.device,
    input_hw: Tuple[int, int],
    batch_size: int = 1,
    warmup: int = 50,
    runs: int = 200,
    use_amp: bool = False,
    channels_last: bool = False,
    num_threads: int = 0,
) -> Dict[str, float]:
    """
    Returns: {"avg_ms": ..., "std_ms": ..., "runs": ..., "warmup": ...}
    """
    if device.type == "cpu" and num_threads and num_threads > 0:
        try:
            torch.set_num_threads(int(num_threads))
        except Exception:
            pass

    net = net.to(device)
    net.eval()

    x = torch.zeros((int(batch_size), 3, int(input_hw[0]), int(input_hw[1])), device=device, dtype=torch.float32)
    if channels_last:
        x = x.contiguous(memory_format=torch.channels_last)
        try:
            net = net.to(memory_format=torch.channels_last)
        except Exception:
            pass

    # Warmup
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        for _ in range(int(warmup)):
            with torch.cuda.amp.autocast(enabled=bool(use_amp)):
                _ = net(x)
        torch.cuda.synchronize()
        times = []

        starter = torch.cuda.Event(enable_timing=True)
        ender   = torch.cuda.Event(enable_timing=True)

        for _ in range(int(runs)):
            starter.record()
            with torch.cuda.amp.autocast(enabled=bool(use_amp)):
                _ = net(x)
            ender.record()
            torch.cuda.synchronize()
            times.append(float(starter.elapsed_time(ender)))  # ms

    else:
        # CPU timing
        for _ in range(int(warmup)):
            _ = net(x)
        times = []
        for _ in range(int(runs)):
            t0 = time.perf_counter()
            _ = net(x)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

    arr = np.array(times, dtype=np.float64) if len(times) else np.array([float("nan")], dtype=np.float64)
    return {
        "avg_ms": float(np.nanmean(arr)),
        "std_ms": float(np.nanstd(arr)),
        "runs": float(runs),
        "warmup": float(warmup),
    }



def load_state_dict_weights_only(path: str, map_location: str):
    """Load a pure state_dict safely.

    Uses torch.load(..., weights_only=True) when supported.
    Cleans non-weight keys like 'mask_values'.
    """

    try:
        state = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=map_location)

    # If checkpoint is wrapped (e.g., {"state_dict": ...})
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    # Remove unwanted metadata keys
    if isinstance(state, dict):
        state.pop("mask_values", None)

    return state




# ----------------------------
# Public aliases (clean API)
# ----------------------------
print_flops_per_conv = _print_flops_per_conv
print_part_compare = _print_part_compare
print_final_summary = _print_final_summary
