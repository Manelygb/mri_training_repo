# friendly_pruning.py
# Unified + safer Taylor channel pruning for Prune_U-Net (Carvana) + "hardware-friendly" channel counts
#
# This file is a drop-in sibling of fixed_pruning.py.
# It keeps the same behavior, logs, tables, and workflow, but adds:
#   (A) Friendly pruning: enforce output-channel counts that are multiples of a user-chosen value
#       (typically 8/16/32) to improve backend efficiency (CUDA/CPU kernels).
#       - Implemented by selecting channels to prune in BLOCKS of `friendly_multiple`
#         per conv layer (baseline channels are typically multiples already).
#   (B) End-of-run inference benchmark on BOTH CPU and GPU (if available):
#       - Average latency over warmup + N runs.
#
# Usage examples:
#   Global friendly pruning:
#     python friendly_pruning.py --scope global --prune_channels 800 --gpu --name "friendly_global_b800"
#
#   Set CUDA-friendly multiple to 16 (common for FP16/TensorCore aligned shapes):
#     python friendly_pruning.py --gpu --friendly 1 --friendly_multiple_gpu 16
#
# Notes:
# - Friendly pruning targets multiples for ALL Conv2d layers except outc.* (kept).
# - If requested prune count is not divisible by the friendly multiple, the script rounds UP
#   to the next multiple (so it prunes at least what you asked for) and logs the adjustment.
#
# Assumptions about prune_utils.Pruner:
# - Attributes: convs, conv_names, ranks (dict[layer_idx]->1D tensor), flops (list), flops_reg (float)
# - Methods: calc_flops(), compute_rank(), pruning(prune_channels), channel_save(path)
# - __init__(net, flops_reg, min_keep_ch=..., min_keep_ratio=...) is supported

import os
import os.path as osp
import io
import json
import random
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
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


# ----------------------------
# CLI
# ----------------------------
def get_args():
    p = OptionParser()

    p.add_option("-n", "--name", dest="name", default="friendly", help="run name")
    p.add_option("-b", "--batch_size", dest="batch_size", default=2, type="int", help="batch size")
    p.add_option("-t", "--taylor_batches", dest="taylor_batches", default=500, type="int", help="mini-batches for Taylor criterion")

    # scope:
    p.add_option("--scope", dest="scope", default="flops_based", help="global|encoder|bottleneck|decoder|custom|pooled_down|flops_based|params_based")

    # global or single-part uses --prune_channels
    p.add_option("-p", "--prune_channels", dest="prune_channels", default=300, type="int", help="channels to remove (global/single-part)")

    # custom budgets
    p.add_option("--prune_encoder", dest="prune_encoder", default=0, type="int", help="custom: encoder budget")
    p.add_option("--prune_bottleneck", dest="prune_bottleneck", default=0, type="int", help="custom: bottleneck budget")
    p.add_option("--prune_decoder", dest="prune_decoder", default=0, type="int", help="custom: decoder budget")
    p.add_option("--prune_down_total", dest="prune_down_total", default=-1, type="int", help="pooled_down: pooled encoder+bottleneck budget (>=0 enables)")

    p.add_option("--include_inc", dest="include_inc", type="int", default=1, help="1 include inc.* in encoder selection, 0 exclude inc.*")

    p.add_option("-g", "--gpu", action="store_true", dest="gpu", default=True, help="use cuda")
    p.add_option("-l", "--load", dest="load", default="weights/unet_carvana_scale0.5_epoch2.pth", help="checkpoint path")
    p.add_option("-c", "--channel_txt", dest="channel_txt", default="model_channels.txt", help="channel txt")
    p.add_option("-s", "--scale", dest="scale", type="float", default=0.5, help="downscale factor")

    p.add_option("-r", "--lr", dest="lr", type="float", default=0.001, help="lr finetune")
    p.add_option("-i", "--iters", dest="iters", type="int", default=1500, help="finetune iters")
    p.add_option("-e", "--epochs", dest="epochs", type="int", default=None, help="finetune epochs")
    p.add_option("-f", "--flops", dest="flops_reg", type="float", default=0.001, help="flops reg")

    p.add_option("--seed", dest="seed", type="int", default=0, help="seed")
    p.add_option("--taylor_shuffle", dest="taylor_shuffle", type="int", default=0, help="shuffle in Taylor batches")
    p.add_option("--data_root", dest="data_root", default="~/work/carvana", help="folder containing train/ and train_masks/")
    p.add_option("--eval_baseline", dest="eval_baseline", type="int", default=1, help="1 eval baseline, 0 skip")
    p.add_option("--baseline_dice", dest="baseline_dice", type="float", default=0.989935, help="cached baseline dice used if baseline eval is skipped")

    p.add_option("--quiet_split", dest="quiet_split", type="int", default=1, help="reduce split logging")
    p.add_option("--print_flops", dest="print_flops", type="int", default=1, help="print flops tables")
    p.add_option("--print_table", dest="print_table", type="int", default=1, help="print final summaries")

    # Taylor mode
    p.add_option("--freeze_bn_taylor", dest="freeze_bn_taylor", type="int", default=1, help="1 freeze BN (eval) during Taylor, 0 keep BN in train")

    # Non-negative ranks
    p.add_option("--rank_transform", dest="rank_transform", default="abs", help="abs|square|none (default abs)")

    # Structural safety
    p.add_option("--min_keep_ch", dest="min_keep_ch", type="int", default=1, help="Min out_channels to keep per conv (default 8).")
    p.add_option("--min_keep_ratio", dest="min_keep_ratio", type="float", default=0.0, help="Min keep ratio per conv (default 0.10).")

    # Candidate overfetch (kept for interface compatibility; friendly pruning ignores it)
    p.add_option("--rank_overfetch", dest="rank_overfetch", type="int", default=5, help="overfetch factor for candidate list")

    # Optional: cap candidate count per layer (kept for interface compatibility; friendly pruning ignores it)
    p.add_option("--max_candidates_per_layer", dest="max_candidates_per_layer", type="int", default=0,
                 help="optional cap on selection candidates per layer; 0 disables")

    # FRIENDLY pruning controls
    p.add_option("--friendly", dest="friendly", type="int", default=1, help="1 enable friendly pruning, 0 disable")
    p.add_option("--friendly_round", dest="friendly_round", default="up", help="up|down: round requested prune to multiple")
    p.add_option("--friendly_multiple_cpu", dest="friendly_multiple_cpu", type="int", default=16, help="friendly multiple for CPU runs")
    p.add_option("--friendly_multiple_gpu", dest="friendly_multiple_gpu", type="int", default=16, help="friendly multiple for CUDA runs")

    # Benchmark controls
    p.add_option("--bench_batch", dest="bench_batch", type="int", default=1, help="benchmark batch size")
    p.add_option("--bench_warmup", dest="bench_warmup", type="int", default=5, help="benchmark warmup iters")
    p.add_option("--bench_runs", dest="bench_runs", type="int", default=50, help="benchmark timed runs")
    p.add_option("--bench_amp", dest="bench_amp", type="int", default=0, help="1 enable AMP for CUDA benchmark")
    p.add_option("--bench_channels_last", dest="bench_channels_last", type="int", default=0, help="1 use channels_last for benchmark")
    p.add_option("--bench_threads", dest="bench_threads", type="int", default=0, help="CPU threads for benchmark (0 = default)")


    p.add_option("--enc_ratio", dest="enc_ratio", type="float", default=0.15,
                 help="Encoder pruning ratio")
    p.add_option("--bn_ratio", dest="bn_ratio", type="float", default=0.45,
                 help="Bottleneck pruning ratio")
    p.add_option("--dec_ratio", dest="dec_ratio", type="float", default=0.40,
                 help="Decoder pruning ratio")

    (opt, _) = p.parse_args()
    return opt


def _resolve_budget_and_requested(args) -> Tuple[Optional[GroupBudget], int, str]:
    scope = str(args.scope).lower().strip()

    if scope == "global":
        return None, int(args.prune_channels), "GLOBAL"

    if scope == "flops_based":
        return None, int(args.prune_channels), "FLOPS BASED"

    if scope == "params_based":
        return None, int(args.prune_channels), "PARAMS BASED"

    if scope in ("encoder", "bottleneck", "decoder"):
        pc = int(args.prune_channels)
        if scope == "encoder":
            b = make_group_budget(prune_encoder=pc, prune_bottleneck=0, prune_decoder=0)
        elif scope == "bottleneck":
            b = make_group_budget(prune_encoder=0, prune_bottleneck=pc, prune_decoder=0)
        else:
            b = make_group_budget(prune_encoder=0, prune_bottleneck=0, prune_decoder=pc)
        return b, int(b.total()), f"SINGLE-PART:{scope.upper()}"

    if scope == "custom":
        pe = int(args.prune_encoder)
        pb = int(args.prune_bottleneck)
        pd = int(args.prune_decoder)
        if pe + pb + pd <= 0:
            raise ValueError("custom scope requires at least one of --prune_encoder/--prune_bottleneck/--prune_decoder > 0")
        b = make_group_budget(prune_encoder=pe, prune_bottleneck=pb, prune_decoder=pd)
        return b, int(b.total()), "CUSTOM (encoder/bottleneck/decoder)"

    if scope == "pooled_down":
        d = int(args.prune_down_total)
        if d < 0:
            raise ValueError("pooled_down scope requires --prune_down_total >= 0")
        pd = int(args.prune_decoder)
        b = make_group_budget(prune_down_total=d, prune_decoder=pd)
        return b, int(b.total()), "POOLED-DOWN (encoder+bottleneck) + decoder"

    raise ValueError(f"Unknown --scope: {args.scope}")


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

    # Resolve budget + total requested pruning
    budget, requested_prune_user, scope_desc = _resolve_budget_and_requested(args)

    # Friendly multiple based on pruning device
    friendly_enabled = bool(int(args.friendly) == 1)
    friendly_multiple = int(args.friendly_multiple_gpu if device.type == "cuda" else args.friendly_multiple_cpu)
    friendly_round = str(args.friendly_round).lower().strip()

    runname = args.name
    save_root = "save_friendly"
    save_path = osp.join(save_root, runname)
    save_dir = get_save_dir(save_path, runname, training=False)
    log = get_logger(save_dir, runname)

    # Compute *effective* requested prune for logging (actual enforcement occurs in pruner)
    requested_prune_effective = requested_prune_user
    if friendly_enabled and friendly_multiple > 1:
        if friendly_round == "down":
            requested_prune_effective = floor_to_multiple(requested_prune_user, friendly_multiple)
        else:
            requested_prune_effective = ceil_to_multiple(requested_prune_user, friendly_multiple)

    log.info(
        "Args: {}".format(
            json.dumps(
                {
                    "scope": args.scope,
                    "scope_desc": scope_desc,
                    "requested_prune_user": int(requested_prune_user),
                    "requested_prune_effective": int(requested_prune_effective),
                    "budget": None if budget is None else {
                        "prune_down_total": budget.prune_down_total,
                        "prune_encoder": budget.prune_encoder,
                        "prune_bottleneck": budget.prune_bottleneck,
                        "prune_decoder": budget.prune_decoder,
                        "total": budget.total(),
                    },
                    "friendly_enabled": int(friendly_enabled),
                    "friendly_multiple": int(friendly_multiple),
                    "friendly_round": friendly_round,
                    "batch_size": args.batch_size,
                    "channel_txt": args.channel_txt,
                    "data_root": data_root,
                    "epochs": args.epochs,
                    "eval_baseline": int(args.eval_baseline),
                    "flops_reg": float(args.flops_reg),
                    "gpu": (device.type == "cuda"),
                    "iters": int(args.iters),
                    "load": args.load,
                    "lr": float(args.lr),
                    "scale": float(args.scale),
                    "seed": int(args.seed),
                    "splitfile": splitfile,
                    "taylor_batches": int(args.taylor_batches),
                    "taylor_shuffle": int(args.taylor_shuffle),
                    "freeze_bn_taylor": int(args.freeze_bn_taylor),
                    "rank_transform": args.rank_transform,
                    "min_keep_ch": int(args.min_keep_ch),
                    "min_keep_ratio": float(args.min_keep_ratio),
                    "rank_overfetch": int(args.rank_overfetch),
                    "include_inc": int(args.include_inc),
                    "max_candidates_per_layer": int(args.max_candidates_per_layer),
                    "bench_batch": int(args.bench_batch),
                    "bench_warmup": int(args.bench_warmup),
                    "bench_runs": int(args.bench_runs),
                    "bench_amp": int(args.bench_amp),
                    "bench_channels_last": int(args.bench_channels_last),
                    "bench_threads": int(args.bench_threads),
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

    # Baseline params/size and per-part params BEFORE pruning
    baseline_param_parts = params_by_part(net)
    baseline_params_total = count_params(net)
    baseline_size_mb = size_mb_from_state_dict(net.state_dict())
    baseline_outchs = conv_out_channels_by_name(net)

    # ----------------------------
    # BASELINE evaluation FIRST
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
    # Build FRIENDLY pruner AFTER baseline eval
    # ----------------------------
    max_cand = int(args.max_candidates_per_layer)
    if max_cand <= 0:
        max_cand = None

    pruner = FriendlyBudgetPruner(
        net,
        float(args.flops_reg),
        scope=("global" if args.scope.lower() == "global" else args.scope.lower()),
        include_inc=bool(int(args.include_inc)),
        budget=(budget if args.scope.lower() in ("custom", "pooled_down") else None),
        rank_transform=str(args.rank_transform),
        rank_overfetch=int(args.rank_overfetch),  # kept for API compatibility
        max_candidates_per_layer=max_cand,        # kept for API compatibility
        min_keep_ch=int(args.min_keep_ch),
        min_keep_ratio=float(args.min_keep_ratio),
        friendly_enabled=friendly_enabled,
        friendly_multiple=int(friendly_multiple),
        friendly_round=friendly_round,
    )

    criterion = nn.CrossEntropyLoss()
    taylor_shuffle = bool(int(args.taylor_shuffle))

    # Taylor mode
    set_model_for_taylor(net, freeze_bn=bool(int(args.freeze_bn_taylor)))

    log.info(f"Taylor criterion: {args.taylor_batches} mini-batches (shuffle={taylor_shuffle})")
    log.info(f"Selection scope: {scope_desc}")
    log.info(f"Requested prune total (user): {requested_prune_user}")
    if friendly_enabled and friendly_multiple > 1:
        log.info(f"Friendly pruning: enabled (multiple={friendly_multiple}, round={friendly_round})")
        log.info(f"Requested prune total (effective): {requested_prune_effective}")

    baseline_flops_named = None
    baseline_flops_sum = None
    input_hw = None

    train_gen = get_imgs_and_masks(train_ids, dir_img, dir_mask, args.scale)

    with tqdm(total=int(args.taylor_batches) * int(args.batch_size)) as progress_bar:
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
                flops_list = pruner.calc_flops()
                names = pruner.conv_names[:-1]  # keep same convention as your scripts
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

    # Debug prints (rank stats + selection distribution)
    pruner.log_rank_stats(log)
    _ = pruner.preview_selection(int(requested_prune_user))  # preview uses friendly effective inside pruner
    pruner.log_selection_distribution(log, pruner._last_selected)

    # Determine the actual prune count we will request from Pruner.pruning()
    actual_prune_to_apply = int(requested_prune_user)
    if friendly_enabled and friendly_multiple > 1:
        # use the pruner's last effective target if available; otherwise use computed effective
        if getattr(pruner, "_effective_prune_last", None) is not None:
            actual_prune_to_apply = int(pruner._effective_prune_last)
        else:
            actual_prune_to_apply = int(requested_prune_effective)

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

    
    pruner.prune_by_parts(int(actual_prune_to_apply), ratios)
    log.info(f"Pruning complete: requested removal={actual_prune_to_apply}")

    save_pruned_pth = osp.join(save_dir, "Pruned.pth")
    save_pruned_txt = osp.join(save_dir, "pruned_channels.txt")
    torch.save(net.state_dict(), save_pruned_pth)
    pruner.channel_save(save_pruned_txt)
    log.info(f"Saved pruned weights : {save_pruned_pth}")
    log.info(f"Saved pruned channels: {save_pruned_txt}")

    # IMPORTANT: drop pruner hooks + clear CUDA cache BEFORE rebuilding/evaluating
    del pruner
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ----------------------------
    # Rebuild pruned net (PRE-FINETUNE) + evaluate
    # ----------------------------
    net_pr = UNet(n_channels=3, n_classes=2, f_channels=save_pruned_txt).to(device)
    net_pr.load_state_dict(torch.load(save_pruned_pth, map_location=device))
    log.info("Rebuilt PRUNED model and loaded weights")

    # Print per-conv out_channels deltas (helps detect layer collapse)
    pruned_outchs = conv_out_channels_by_name(net_pr)
    log.info("=" * 92)
    log.info("OUT_CHANNELS PER CONV (Baseline vs Pruned) — collapse detector")
    log.info("=" * 92)
    for k in sorted(set(baseline_outchs.keys()) & set(pruned_outchs.keys())):
        bch = baseline_outchs[k]
        pch = pruned_outchs[k]
        log.info(f"  - {k:40s}: {bch:4d} -> {pch:4d}   (delta {pch-bch:+d})")
    log.info("=" * 92)

    # Evaluate before finetune
    val_gen_pr = get_imgs_and_masks(val_ids, dir_img, dir_mask, args.scale)
    pruned_dice_preft = eval_net(net_pr, val_gen_pr, len(val_ids), (device.type == "cuda"), args.batch_size)
    log.info(f"PRUNED (pre-finetune) Dice: {pruned_dice_preft:.6f}")

    # ----------------------------
    # Finetune
    # ----------------------------
    optimizer = optim.SGD(net_pr.parameters(), lr=float(args.lr), momentum=0.9, weight_decay=0.0005)

    save_finetuned = osp.join(save_dir, "Finetuned.pth")
    finetune(
        net_pr,
        optimizer,
        criterion,
        train_ids,
        log,
        save_finetuned,
        int(args.iters),
        args.epochs,
        int(args.batch_size),
        (device.type == "cuda"),
        float(args.scale),
    )

    # ----------------------------
    # Evaluate pruned+finetuned
    # ----------------------------
    val_gen2 = get_imgs_and_masks(val_ids, dir_img, dir_mask, args.scale)
    pruned_dice = eval_net(net_pr, val_gen2, len(val_ids), (device.type == "cuda"), args.batch_size)
    log.info(f"PRUNED+FINETUNED Dice: {pruned_dice:.6f}")

    # ----------------------------
    # FLOPs after pruning (hook-based, no gradients)
    # ----------------------------
    x_dummy = torch.zeros((1, 3, input_hw[0], input_hw[1]), device=device, dtype=torch.float32)
    pruned_flops_named = conv_flops_named_no_grad(net_pr, x_dummy)

    # Match baseline convention (exclude outc.conv from totals)
    pruned_flops_named_no_outc = [(n, f) for (n, f) in pruned_flops_named if n != "outc.conv"]
    pruned_flops_sum = int(sum(f for _, f in pruned_flops_named_no_outc))

    if int(args.print_flops) == 1:
        _print_flops_per_conv(log, "FLOPs after pruning (per conv):", pruned_flops_named_no_outc)

    # ----------------------------
    # Collect final metrics
    # ----------------------------
    pruned_params_total = count_params(net_pr)
    pruned_size_mb = size_mb_from_state_dict(net_pr.state_dict())

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

        base_flops_parts = summarize_flops_by_part(baseline_flops_named)  # baseline excludes outc
        pr_flops_parts = summarize_flops_by_part(pruned_flops_named_no_outc)
        _print_part_compare(log, "PER-PART FLOPs (Baseline vs Pruned+Finetuned)", base_flops_parts, pr_flops_parts, unit="flops")

        pr_param_parts = params_by_part(net_pr)
        _print_part_compare(log, "PER-PART Params (Baseline vs Pruned+Finetuned)", baseline_param_parts, pr_param_parts, unit="params")

        # Aggregated groups
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

        params_base_by_group = {g: _sum_keys(baseline_param_parts, ks) for g, ks in GROUPS.items()}
        params_pruned_by_group = {g: _sum_keys(pr_param_parts, ks) for g, ks in GROUPS.items()}

        flops_base_by_group = {g: _sum_keys(base_flops_parts, ks) for g, ks in GROUPS.items()}
        flops_pruned_by_group = {g: _sum_keys(pr_flops_parts, ks) for g, ks in GROUPS.items()}

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

    # Always print the pre-finetune dice (important for diagnosis)
    log.info("=" * 92)
    log.info("DIAGNOSIS CHECKPOINT")
    log.info("=" * 92)
    log.info(f"Baseline dice           : {float(baseline_dice):.6f}")
    log.info(f"Pruned pre-finetune dice: {float(pruned_dice_preft):.6f}")
    log.info(f"Pruned finetuned dice   : {float(pruned_dice):.6f}")
    log.info("=" * 92)

    # ----------------------------
    # INFERENCE BENCHMARK (CPU + GPU) — BASELINE vs PRUNED+FINETUNED
    # ----------------------------
    log.info("=" * 92)
    log.info("INFERENCE BENCHMARK (Average latency) — BASELINE vs PRUNED+FINETUNED")
    log.info("=" * 92)

    bench_batch   = int(args.bench_batch)
    bench_warmup  = int(args.bench_warmup)
    bench_runs    = int(args.bench_runs)
    bench_amp     = bool(int(args.bench_amp) == 1)
    bench_cl      = bool(int(args.bench_channels_last) == 1)
    bench_threads = int(args.bench_threads)

    # ---- safe load to avoid torch.load warning (weights_only=True) ----
    def _load_state_dict_weights_only(path: str, map_location: str):
        """
        Avoids FutureWarning by using weights_only=True when possible.
        Assumes checkpoints are pure state_dict (as saved by torch.save(net.state_dict(), ...)).
        If you ever use wrapped checkpoints, extend this helper to extract keys.
        """
        try:
            return torch.load(path, map_location=map_location, weights_only=True)
        except TypeError:
            # older torch versions without weights_only
            return torch.load(path, map_location=map_location)

    # ----------------------------
    # CPU benchmark
    # ----------------------------
    # Baseline on CPU
    net_base_cpu = UNet(n_channels=3, n_classes=2, f_channels=channel_txt_to_use).cpu()
    net_base_cpu.load_state_dict(_load_state_dict_weights_only(args.load, "cpu"), strict=True)
    base_cpu_stats = benchmark_inference_avg_ms(
        net_base_cpu,
        device=torch.device("cpu"),
        input_hw=input_hw,
        batch_size=bench_batch,
        warmup=bench_warmup,
        runs=bench_runs,
        use_amp=False,
        channels_last=bench_cl,
        num_threads=bench_threads,
    )

    # Pruned+Finetuned on CPU
    net_pr_cpu = UNet(n_channels=3, n_classes=2, f_channels=save_pruned_txt).cpu()
    net_pr_cpu.load_state_dict(_load_state_dict_weights_only(save_finetuned, "cpu"), strict=True)
    pr_cpu_stats = benchmark_inference_avg_ms(
        net_pr_cpu,
        device=torch.device("cpu"),
        input_hw=input_hw,
        batch_size=bench_batch,
        warmup=bench_warmup,
        runs=bench_runs,
        use_amp=False,
        channels_last=bench_cl,
        num_threads=bench_threads,
    )

    log.info(
        f"CPU  BASE: avg={base_cpu_stats['avg_ms']:.3f} ms  std={base_cpu_stats['std_ms']:.3f} ms  "
        f"(batch={bench_batch}, warmup={bench_warmup}, runs={bench_runs}, channels_last={int(bench_cl)}, "
        f"threads={bench_threads if bench_threads>0 else 'default'})"
    )
    log.info(
        f"CPU  PRUN: avg={pr_cpu_stats['avg_ms']:.3f} ms  std={pr_cpu_stats['std_ms']:.3f} ms  "
        f"(batch={bench_batch}, warmup={bench_warmup}, runs={bench_runs}, channels_last={int(bench_cl)}, "
        f"threads={bench_threads if bench_threads>0 else 'default'})"
    )
    log.info(
        f"CPU  speedup (BASE/PRUN) = {(base_cpu_stats['avg_ms'] / pr_cpu_stats['avg_ms']):.3f}×"
        if pr_cpu_stats["avg_ms"] > 0 else "CPU  speedup (BASE/PRUN) = nan"
    )

    # ----------------------------
    # GPU benchmark (if available)
    # ----------------------------
    if torch.cuda.is_available():
        # Baseline on GPU
        net_base_gpu = UNet(n_channels=3, n_classes=2, f_channels=channel_txt_to_use).cuda()
        net_base_gpu.load_state_dict(_load_state_dict_weights_only(args.load, "cuda"), strict=True)
        base_gpu_stats = benchmark_inference_avg_ms(
            net_base_gpu,
            device=torch.device("cuda"),
            input_hw=input_hw,
            batch_size=bench_batch,
            warmup=bench_warmup,
            runs=bench_runs,
            use_amp=bench_amp,
            channels_last=bench_cl,
            num_threads=0,
        )

        # Pruned+Finetuned on GPU
        net_pr_gpu = UNet(n_channels=3, n_classes=2, f_channels=save_pruned_txt).cuda()
        net_pr_gpu.load_state_dict(_load_state_dict_weights_only(save_finetuned, "cuda"), strict=True)
        pr_gpu_stats = benchmark_inference_avg_ms(
            net_pr_gpu,
            device=torch.device("cuda"),
            input_hw=input_hw,
            batch_size=bench_batch,
            warmup=bench_warmup,
            runs=bench_runs,
            use_amp=bench_amp,
            channels_last=bench_cl,
            num_threads=0,
        )

        log.info(
            f"CUDA BASE: avg={base_gpu_stats['avg_ms']:.3f} ms  std={base_gpu_stats['std_ms']:.3f} ms  "
            f"(batch={bench_batch}, warmup={bench_warmup}, runs={bench_runs}, amp={int(bench_amp)}, channels_last={int(bench_cl)})"
        )
        log.info(
            f"CUDA PRUN: avg={pr_gpu_stats['avg_ms']:.3f} ms  std={pr_gpu_stats['std_ms']:.3f} ms  "
            f"(batch={bench_batch}, warmup={bench_warmup}, runs={bench_runs}, amp={int(bench_amp)}, channels_last={int(bench_cl)})"
        )
        log.info(
            f"CUDA speedup (BASE/PRUN) = {(base_gpu_stats['avg_ms'] / pr_gpu_stats['avg_ms']):.3f}×"
            if pr_gpu_stats["avg_ms"] > 0 else "CUDA speedup (BASE/PRUN) = nan"
        )
    else:
        log.info("CUDA : not available on this machine")

    log.info("=" * 92)

