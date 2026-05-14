"""
final_pruning.py

Clean orchestration script for Taylor channel pruning (global / per-part / custom budgets)
with optional "friendly" channel alignment and optional CPU/GPU benchmarking.

All heavy logic lives in final_pruning_utils.py.
"""
from __future__ import annotations

import os
import os.path as osp
import json
import argparse

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from finetune import finetune
from eval import eval_net
from unet import UNet
from utils import (
    get_ids,
    split_train_val,
    get_imgs_and_masks,
    batch,
    get_logger,
    get_save_dir,
)

from final_pruning_utils import (
    seed_all,
    count_params,
    size_mb_from_state_dict,
    params_by_part,
    summarize_flops_by_part,
    conv_out_channels_by_name,
    conv_flops_named_no_grad,
    compare_state_dicts_compact,
    set_model_for_taylor,
    resolve_budget_and_requested,
    ceil_to_multiple,
    floor_to_multiple,
    FriendlyBudgetPruner,
    benchmark_inference_avg_ms,
    load_state_dict_weights_only,
    print_flops_per_conv,
    print_part_compare,
    print_final_summary,
    compute_targets_from_pct,
    take_dep_snapshot,
    dep_pruned_counts,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # Identity / paths
    p.add_argument("-n", "--name", default="final_mri", help="run name (folder name)")
    p.add_argument("--save_root", default="save_mri", help="root output folder")
    p.add_argument("--data_root", default="dataset_split", help="folder containing train/ and train_masks/")
    p.add_argument("--splitfile", default="dataset_split/split.json", help="train/val split json path")
    p.add_argument("-l", "--load", default="mri_checkpoint_epoch100.pth", help="checkpoint path (state_dict)")
    p.add_argument("-c", "--channel_txt", default="model_channels.txt", help="channel txt (baseline channels)")

    # Device
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="training/pruning device")
    p.add_argument("-g", "--gpu", action="store_true", help="alias for --device cuda")

    # Data / eval
    p.add_argument("-s", "--scale", type=float, default=0.5, help="downscale factor")
    p.add_argument("-b", "--batch_size", type=int, default=2, help="batch size")
    p.add_argument("--quiet_split", type=int, default=0, help="1 to silence split logging")

    # Taylor criterion
    p.add_argument("-t", "--taylor_batches", type=int, default=500, help="mini-batches for Taylor criterion")
    p.add_argument("--taylor_shuffle", type=int, default=0, help="shuffle in Taylor batches (0/1)")
    p.add_argument("--freeze_bn_taylor", type=int, default=1, help="1 freeze BN during Taylor, 0 keep BN train")
    p.add_argument("--rank_transform", default="abs", help="abs|square|none (default abs)")
    p.add_argument("--rank_overfetch", type=int, default=5, help="overfetch factor (kept for compatibility)")
    p.add_argument("--max_candidates_per_layer", type=int, default=0, help="kept for compatibility; 0 disables")

    # Pruning scope / budgets
    p.add_argument("--scope", default="global", help="global|encoder|bottleneck|decoder|custom|pooled_down|custom_seq")
    p.add_argument("-p", "--prune_channels", type=int, default=300, help="channels to remove (global/single-part)")
    # for scope=custom
    p.add_argument("--prune_encoder", type=int, default=0, help="custom: encoder budget")
    p.add_argument("--prune_bottleneck", type=int, default=0, help="custom: bottleneck budget")
    p.add_argument("--prune_decoder", type=int, default=0, help="custom: decoder budget")
    # for scope = pooled_down
    p.add_argument("--prune_down_total", type=int, default=-1, help="pooled_down: pooled encoder+bottleneck budget (>=0 enables)")
    # for scope = custom_seq (prct args)
    p.add_argument("--prune_encoder_pct", type=float, default=0.0, help="custom_seq: encoder share (e.g., 20 for 20%)")
    p.add_argument("--prune_bottleneck_pct", type=float, default=0.0, help="custom_seq: bottleneck share (e.g., 60 for 60%)")
    p.add_argument("--prune_decoder_pct", type=float, default=0.0, help="custom_seq: decoder share (e.g., 20 for 20%)")
    # more args
    p.add_argument("--include_inc", type=int, default=1, help="1 include inc.* in encoder selection, 0 exclude inc.*")
    p.add_argument("--min_keep_ch", type=int, default=1, help="minimum out_channels to keep per conv")
    p.add_argument("--min_keep_ratio", type=float, default=0.0, help="minimum keep ratio per conv (0 disables)")

    # Friendly pruning controls
    p.add_argument("--friendly", type=int, default=1, help="1 enable friendly pruning, 0 disable")
    p.add_argument("--friendly_round", default="up", choices=["up", "down"], help="round requested prune to multiple")
    p.add_argument("--friendly_multiple_cpu", type=int, default=16, help="friendly multiple for CPU runs")
    p.add_argument("--friendly_multiple_gpu", type=int, default=16, help="friendly multiple for CUDA runs")

    # Regularization
    p.add_argument("-f", "--flops", dest="flops_reg", type=float, default=0.001, help="FLOPs regularizer weight")

    # Finetuning
    p.add_argument("-r", "--lr", type=float, default=0.0025, help="finetune lr")
    p.add_argument("-i", "--iters", type=int, default=6000, help="finetune iterations")
    p.add_argument("-e", "--epochs", type=int, default=None, help="(compat) epochs, 0 uses iters only")

    # Baseline eval
    p.add_argument("--eval_baseline", type=int, default=1, help="1 eval baseline, 0 skip")
    p.add_argument("--baseline_dice", type=float, default=0.88795, help="cached baseline dice used if baseline eval is skipped")

    # Diagnostics
    p.add_argument("--print_flops", type=int, default=0, help="1 print per-conv FLOPs tables")

    # Benchmarking
    p.add_argument("--print_table", type=int, default=1, help="print final summaries (1=yes, 0=no)")
    p.add_argument("--bench", action="store_true", help="compute CPU/GPU inference time and report speedup (default: off)")
    p.add_argument("--bench_batch", type=int, default=1, help="benchmark batch size")
    p.add_argument("--bench_warmup", type=int, default=5, help="benchmark warmup iters")
    p.add_argument("--bench_runs", type=int, default=50, help="benchmark timed runs")
    p.add_argument("--bench_amp", type=int, default=0, help="1 enable AMP for CUDA benchmark")
    p.add_argument("--bench_channels_last", type=int, default=0, help="1 use channels_last for benchmark")
    p.add_argument("--bench_threads", type=int, default=0, help="CPU threads for benchmark (0 = default)")

    # Repro
    p.add_argument("--seed", type=int, default=42, help="random seed")

    args = p.parse_args()
    if args.gpu:
        args.device = "cuda"
    return args


def _select_device(args: argparse.Namespace) -> torch.device:
    if args.device == "cpu":
        return torch.device("cpu")
    if args.device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()

    data_root = osp.expanduser(args.data_root)

    train_root = osp.join(data_root, "train")
    val_root   = osp.join(data_root, "val")
    
    dir_img = osp.join(train_root, "imgs") + "/"
    dir_mask = osp.join(train_root, "masks") + "/"
    
    dir_img_val = osp.join(val_root, "imgs") + "/"
    dir_mask_val = osp.join(val_root, "masks") + "/"
    
    splitfile = osp.expanduser(args.splitfile)

    if not osp.isdir(dir_img):
        raise FileNotFoundError(f"imgs/ not found: {dir_img} (use --data_root)")
    if not osp.isdir(dir_mask):
        raise FileNotFoundError(f"masks/ not found: {dir_mask} (use --data_root)")

    device = _select_device(args)
    seed_all(args.seed, use_cuda=(device.type == "cuda"))

    # Resolve scope + budgets
    prune_down_total = None if int(args.prune_down_total) < 0 else int(args.prune_down_total)
    
    if str(args.scope).lower().strip() == "custom_seq":
        # total budget is prune_channels; per-group targets are computed later from *_pct
        budget, requested_prune_user, scope_desc = None, int(args.prune_channels), "CUSTOM_SEQ (total + pct, sequential)"
    else:
        budget, requested_prune_user, scope_desc = resolve_budget_and_requested(
            scope=args.scope,
            prune_channels=int(args.prune_channels),
            prune_encoder=int(args.prune_encoder),
            prune_bottleneck=int(args.prune_bottleneck),
            prune_decoder=int(args.prune_decoder),
            prune_down_total=prune_down_total,
        )

    # Friendly settings
    friendly_enabled = bool(int(args.friendly) == 1)
    friendly_multiple = int(args.friendly_multiple_gpu if device.type == "cuda" else args.friendly_multiple_cpu)
    friendly_round = str(args.friendly_round).lower().strip()

    # Output folder + logger
    runname = str(args.name)
    save_path = osp.join(str(args.save_root), runname)
    save_dir = get_save_dir(save_path, runname, training=False)
    log = get_logger(save_dir, runname)

    # Effective requested prune (for logging only; enforcement happens in pruner)
    requested_prune_effective = int(requested_prune_user)
    if friendly_enabled and friendly_multiple > 1:
        if friendly_round == "down":
            requested_prune_effective = floor_to_multiple(requested_prune_user, friendly_multiple)
        else:
            requested_prune_effective = ceil_to_multiple(requested_prune_user, friendly_multiple)

    log.info("Args: {}".format(json.dumps({
        "name": runname,
        "save_dir": save_dir,
        "device": device.type,
        "data_root": data_root,
        "scale": float(args.scale),
        "batch_size": int(args.batch_size),
        "taylor_batches": int(args.taylor_batches),
        "taylor_shuffle": int(args.taylor_shuffle),
        "freeze_bn_taylor": int(args.freeze_bn_taylor),
        "seed": int(args.seed),
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
        "min_keep_ch": int(args.min_keep_ch),
        "min_keep_ratio": float(args.min_keep_ratio),
        "flops_reg": float(args.flops_reg),
        "finetune_iters": int(args.iters),
        "finetune_lr": float(args.lr),
        "eval_baseline": int(args.eval_baseline),
        "bench": bool(args.bench),
    }, indent=2)))

    # ----------------------------
    # Train/val split
    # ----------------------------
    #if not osp.exists(splitfile):
    #    os.makedirs(osp.dirname(splitfile), exist_ok=True)
    #    ids = get_ids(dir_img)
    #    iddataset = split_train_val(ids, 0.2, splitfile)
    #    if int(args.quiet_split) != 1:
    #        log.info(f"New split dataset -> {splitfile}")
    #else:
    #    with open(splitfile) as f:
    #        iddataset = json.load(f)
    #    if int(args.quiet_split) != 1:
    #        log.info(f"Load split dataset -> {splitfile}")

    #train_ids = iddataset["train"]
    #val_ids = iddataset["val"]
    train_ids = get_ids(dir_img)
    val_ids = get_ids(dir_img_val)
    
    log.info(f"Training samples: {len(train_ids)}")
    log.info(f"Validation samples: {len(val_ids)}")

    # ----------------------------
    # Model init + load
    # ----------------------------
    channel_txt = osp.expanduser(args.channel_txt)
    if not osp.exists(channel_txt):
        raise FileNotFoundError(f"channel_txt not found: {channel_txt} (use --channel_txt)")
    net = UNet(n_channels=3, n_classes=2, f_channels=channel_txt).to(device)
    log.info(f"Built model using channel file: {channel_txt}")

    if args.load and osp.exists(args.load):
        ckpt_sd = load_state_dict_weights_only(args.load, map_location=device)
        common, mismatch, missing, unexpected = compare_state_dicts_compact(ckpt_sd, net.state_dict())
        log.info(f"Checkpoint key/shape check: common={common}, mismatch={mismatch}, missing={missing}, unexpected={unexpected}")
        net.load_state_dict(ckpt_sd)
        log.info(f"Loaded checkpoint: {args.load}")
    else:
        log.info(f"Checkpoint not found: {args.load} (continuing)")

    # ----------------------------
    # Baseline metrics
    # ----------------------------
    baseline_params_total = count_params(net)
    baseline_size_mb = size_mb_from_state_dict(net.state_dict())
    baseline_param_parts = params_by_part(net)

    if int(args.eval_baseline) == 1:
        log.info("Evaluating BASELINE model...")
        val_gen = get_imgs_and_masks(val_ids, dir_img_val, dir_mask_val, args.scale)
        #val_gen = get_imgs_and_masks(val_ids, dir_img, dir_mask, args.scale)
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
    # Build pruner
    # ----------------------------
    max_cand = int(args.max_candidates_per_layer)
    if max_cand <= 0:
        max_cand = None

    pruner = FriendlyBudgetPruner(
        net,
        float(args.flops_reg),
        scope=args.scope.lower(),
        include_inc=bool(int(args.include_inc)),
        budget=budget,
        rank_transform=str(args.rank_transform),
        rank_overfetch=int(args.rank_overfetch),
        max_candidates_per_layer=max_cand,
        min_keep_ch=int(args.min_keep_ch),
        min_keep_ratio=float(args.min_keep_ratio),
        friendly_enabled=friendly_enabled,
        friendly_multiple=int(friendly_multiple),
        friendly_round=friendly_round,
    )

    criterion = nn.CrossEntropyLoss()
    taylor_shuffle = bool(int(args.taylor_shuffle))
    
    baseline_flops_named = None
    baseline_flops_sum = None
    input_hw = None
    
    def _run_taylor_batches(stage_label: str):
        nonlocal baseline_flops_named, baseline_flops_sum, input_hw
    
        pruner.clear_cache()
        pruner.clear_rank()
        set_model_for_taylor(net, freeze_bn=bool(int(args.freeze_bn_taylor)))
    
        train_gen_local = get_imgs_and_masks(train_ids, dir_img, dir_mask, args.scale)
    
        with tqdm(total=int(args.taylor_batches) * int(args.batch_size), desc=f"Taylor ({stage_label})") as progress_bar:
            for i, b0 in enumerate(batch(train_gen_local, args.batch_size, shuffle=taylor_shuffle)):
                net.zero_grad(set_to_none=True)
    
                imgs_np = np.array([x[0] for x in b0], dtype=np.float32)
                masks_np = np.array([x[1] for x in b0])
    
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
                    flops_list = pruner.calc_flops()  # updates pruner.flops for flops-regularized ranking
                    if baseline_flops_named is None:
                        names = pruner.conv_names[:-1]
                        baseline_flops_named = list(zip(names, flops_list))
                        baseline_flops_sum = int(sum(flops_list))
                        if int(args.print_flops) == 1:
                            print_flops_per_conv(log, "FLOPs before pruning (per conv):", baseline_flops_named)
    
                pruner.compute_rank()
                progress_bar.update(len(b0))
    
                if (i + 1) >= int(args.taylor_batches):
                    log.info(f"Finished Taylor criterion computation ({stage_label})")
                    break
    
    if str(args.scope).lower().strip() == "custom_seq":
        # total -> per-group targets (encoder first, then bottleneck, then decoder),
        # with dependency-aware remaining budgets after each step.
        target_enc, target_bot, target_dec = compute_targets_from_pct(
            int(args.prune_channels),
            float(args.prune_encoder_pct),
            float(args.prune_bottleneck_pct),
            float(args.prune_decoder_pct),
        )
    
        log.info("=" * 92)
        log.info("CUSTOM_SEQ TARGETS")
        log.info("=" * 92)
        log.info(f"Total target: {int(args.prune_channels)} | encoder={target_enc} | bottleneck={target_bot} | decoder={target_dec}")
    
        base_dep = take_dep_snapshot(net)
    
        # ----------------------------
        # Stage 1 — Encoder
        # ----------------------------
        if target_enc > 0:
            pruner.scope = "encoder"
            pruner.budget = None
            _run_taylor_batches("encoder")
            pruner.log_rank_stats(log)
            _ = pruner.preview_selection(int(target_enc))
            pruner.log_selection_distribution(log, pruner._last_selected)
    
            to_apply = int(target_enc)
            if friendly_enabled and int(friendly_multiple) > 1 and getattr(pruner, "_effective_prune_last", None) is not None:
                to_apply = int(pruner._effective_prune_last)
    
            pruner.pruning(int(to_apply))
            log.info(f"Stage encoder: requested={target_enc}, applied={to_apply} (friendly-adjusted={to_apply != target_enc})")
        else:
            log.info("Stage encoder: skipped (target=0)")
    
        # dependency after encoder (cumulative vs baseline)
        dep = dep_pruned_counts(base_dep, take_dep_snapshot(net))
        bot_dep = int(dep.get("bottleneck_dep", 0))
        dec_dep = int(dep.get("decoder_dep", 0))
    
        # ----------------------------
        # Stage 2 — Bottleneck
        # ----------------------------
        remaining_bot = max(0, int(target_bot) - int(bot_dep))
        if remaining_bot <= 0:
            if target_bot > 0 and bot_dep > target_bot:
                log.info(f"Stage bottleneck: skipped (dependency pruning exceeded target: dep={bot_dep} > target={target_bot})")
            else:
                log.info("Stage bottleneck: skipped (target satisfied by dependencies)")
        else:
            pruner.scope = "bottleneck"
            pruner.budget = None
            _run_taylor_batches("bottleneck")
            pruner.log_rank_stats(log)
            _ = pruner.preview_selection(int(remaining_bot))
            pruner.log_selection_distribution(log, pruner._last_selected)
    
            to_apply = int(remaining_bot)
            if friendly_enabled and int(friendly_multiple) > 1 and getattr(pruner, "_effective_prune_last", None) is not None:
                to_apply = int(pruner._effective_prune_last)
    
            pruner.pruning(int(to_apply))
            log.info(f"Stage bottleneck: requested_remaining={remaining_bot}, applied={to_apply} (friendly-adjusted={to_apply != remaining_bot})")
    
        # dependency after bottleneck (cumulative vs baseline)
        dep = dep_pruned_counts(base_dep, take_dep_snapshot(net))
        dec_dep = int(dep.get("decoder_dep", 0))
    
        # ----------------------------
        # Stage 3 — Decoder
        # ----------------------------
        remaining_dec = max(0, int(target_dec) - int(dec_dep))
        if remaining_dec <= 0:
            if target_dec > 0 and dec_dep > target_dec:
                log.info(f"Stage decoder: skipped (dependency pruning exceeded target: dep={dec_dep} > target={target_dec})")
            else:
                log.info("Stage decoder: skipped (target satisfied by dependencies)")
        else:
            pruner.scope = "decoder"
            pruner.budget = None
            _run_taylor_batches("decoder")
            pruner.log_rank_stats(log)
            _ = pruner.preview_selection(int(remaining_dec))
            pruner.log_selection_distribution(log, pruner._last_selected)
    
            to_apply = int(remaining_dec)
            if friendly_enabled and int(friendly_multiple) > 1 and getattr(pruner, "_effective_prune_last", None) is not None:
                to_apply = int(pruner._effective_prune_last)
    
            pruner.pruning(int(to_apply))
            log.info(f"Stage decoder: requested_remaining={remaining_dec}, applied={to_apply} (friendly-adjusted={to_apply != remaining_dec})")
    
        if input_hw is None:
            raise RuntimeError("Could not infer input HW (train set empty?)")
    
        baseline_metrics["flops_g"] = float(baseline_flops_sum) / 1e9
    
        # For downstream logging, treat this as the total requested target.
        actual_prune_to_apply = int(target_enc + remaining_bot + remaining_dec)
    
    else:
        # --- original single-pass behavior (unchanged) ---
        set_model_for_taylor(net, freeze_bn=bool(int(args.freeze_bn_taylor)))
    
        train_gen = get_imgs_and_masks(train_ids, dir_img, dir_mask, args.scale)
    
        with tqdm(total=int(args.taylor_batches) * int(args.batch_size)) as progress_bar:
            for i, b0 in enumerate(batch(train_gen, args.batch_size, shuffle=taylor_shuffle)):
                net.zero_grad(set_to_none=True)
    
                imgs_np = np.array([x[0] for x in b0], dtype=np.float32)
                masks_np = np.array([x[1] for x in b0])
    
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
                    names = pruner.conv_names[:-1]
                    baseline_flops_named = list(zip(names, flops_list))
                    baseline_flops_sum = int(sum(flops_list))
                    if int(args.print_flops) == 1:
                        print_flops_per_conv(log, "FLOPs before pruning (per conv):", baseline_flops_named)
    
                pruner.compute_rank()
                progress_bar.update(len(b0))
    
                if (i + 1) >= int(args.taylor_batches):
                    log.info("Finished Taylor criterion computation")
                    break
    
        if input_hw is None:
            raise RuntimeError("Could not infer input HW (train set empty?)")
    
        baseline_metrics["flops_g"] = float(baseline_flops_sum) / 1e9
    
        pruner.log_rank_stats(log)
        _ = pruner.preview_selection(int(requested_prune_user))
        pruner.log_selection_distribution(log, pruner._last_selected)
    
        actual_prune_to_apply = int(requested_prune_user)
        if friendly_enabled and friendly_multiple > 1:
            if getattr(pruner, "_effective_prune_last", None) is not None:
                actual_prune_to_apply = int(pruner._effective_prune_last)
            else:
                actual_prune_to_apply = int(requested_prune_effective)






    
    # ----------------------------
    # Prune & save
    # ----------------------------
    pruner.pruning(int(actual_prune_to_apply))
    log.info(f"Pruning complete: requested removal={actual_prune_to_apply}")

    save_pruned_pth = osp.join(save_dir, "Pruned.pth")
    save_pruned_txt = osp.join(save_dir, "pruned_channels.txt")
    torch.save(net.state_dict(), save_pruned_pth)
    pruner.channel_save(save_pruned_txt)
    log.info(f"Saved pruned weights : {save_pruned_pth}")
    log.info(f"Saved pruned channels: {save_pruned_txt}")

    # Rebuild pruned-architecture UNet from pruned_channels.txt (matches original workflow)
    net_pr = UNet(n_channels=3, n_classes=2, f_channels=save_pruned_txt).to(device)
    net_pr.load_state_dict(load_state_dict_weights_only(save_pruned_pth, map_location=device))
    net = net_pr  # continue with pruned-architecture model

    # ----------------------------
    # Post-prune metrics (params/flops)
    # ----------------------------
    pruned_params_total = count_params(net)
    pruned_size_mb = size_mb_from_state_dict(net.state_dict())
    pruned_param_parts = params_by_part(net)

    # hook-based flops after pruning (no grads)
    dummy = torch.zeros((1, 3, input_hw[0], input_hw[1]), device=device, dtype=torch.float32)
    pruned_flops_named = conv_flops_named_no_grad(net, dummy)
    pruned_flops_named_no_outc = [(n, f) for (n, f) in pruned_flops_named if n != "outc.conv"]
    pruned_flops_sum = int(sum(v for _, v in pruned_flops_named_no_outc))

    pruned_metrics = {
        "dice": float("nan"),  # will fill after finetune eval
        "params": int(pruned_params_total),
        "size_mb": float(pruned_size_mb),
        "flops_g": float(pruned_flops_sum) / 1e9,
    }

    if int(args.print_flops) == 1:
        print_flops_per_conv(log, "FLOPs after pruning (per conv):", pruned_flops_named_no_outc)
    
    # ----------------------------
    # Finetune
    # ----------------------------
    log.info("=" * 92)
    log.info("FINETUNE")
    log.info("=" * 92)

    optimizer = optim.SGD(net.parameters(), lr=float(args.lr), momentum=0.9)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #optimizer,
    #T_max=5000,          # match your iters
    #eta_min=1e-5         # small final LR
    #)
   

    save_finetuned = osp.join(save_dir, "Finetuned.pth")
    finetune(
        net,
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
    log.info(f"Saved finetuned weights: {save_finetuned}")

    # Eval finetuned
    log.info("Evaluating FINETUNED model...")
    #val_gen_eval = get_imgs_and_masks(val_ids, dir_img, dir_mask, args.scale)
    val_gen_eval = get_imgs_and_masks(val_ids, dir_img_val, dir_mask_val, args.scale)

    finetuned_dice = eval_net(net, val_gen_eval, len(val_ids), (device.type == "cuda"), args.batch_size)
    seed_all(args.seed, use_cuda=(device.type == "cuda"))
    pruned_metrics["dice"] = float(finetuned_dice)
    log.info(f"FINETUNED Dice: {finetuned_dice:.6f}")

    # ----------------------------
    # Reporting
    # ----------------------------
    log.info("=" * 92)
    log.info("SUMMARY")
    log.info("=" * 92)

    if int(args.print_table) == 1:
        print_final_summary(log, baseline_metrics, pruned_metrics)
    
        # Per-part FLOPs (match old behavior)
        base_flops_parts = summarize_flops_by_part(baseline_flops_named)
        pr_flops_parts   = summarize_flops_by_part(pruned_flops_named_no_outc)
        print_part_compare(
            log,
            "PER-PART FLOPs (Baseline vs Pruned+Finetuned)",
            base_flops_parts,
            pr_flops_parts,
            unit="flops",
        )
    
        # Per-part Params
        print_part_compare(
            log,
            "PER-PART Params (Baseline vs Pruned+Finetuned)",
            baseline_param_parts,
            pruned_param_parts,
            unit="params",
        )
    
        # Aggregated groups (same grouping as before)
        GROUPS = {
            "ENCODER (inc+down1+down2+down3)": ["inc", "down1", "down2", "down3"],
            "BOTTLENECK (down4)":             ["down4"],
            "DECODER (up1+up2+up3+up4+outc)":  ["up1", "up2", "up3", "up4", "outc"],
        }
        
        def _sum_keys(d: dict, keys: list) -> float:
            return float(sum(d.get(k, 0) for k in keys))
        
        def _reduction_pct(b: float, p: float) -> float:
            # percent reduction from baseline -> pruned
            return 0.0 if b == 0 else (1.0 - (p / b)) * 100.0
        
        log.info("=" * 120)
        log.info("AGGREGATED GROUPS")
        log.info("=" * 120)
        log.info(
            f"{'Group':34s} | {'Params (B)':>12s} | {'Params (P)':>12s} | {'Param ↓%':>9s} | "
            f"{'FLOPs (B)':>12s} | {'FLOPs (P)':>12s} | {'FLOPs ↓%':>9s}"
        )
        log.info("-" * 120)
        
        for gname, keys in GROUPS.items():
            pb = _sum_keys(baseline_param_parts, keys)
            pp = _sum_keys(pruned_param_parts, keys)
            fb = _sum_keys(base_flops_parts, keys)
            fp = _sum_keys(pr_flops_parts, keys)
        
            pr_pct = _reduction_pct(pb, pp)
            fr_pct = _reduction_pct(fb, fp)
        
            log.info(
                f"{gname:34s} | {int(pb):12d} | {int(pp):12d} | {pr_pct:8.2f}% | "
                f"{(fb/1e9):12.3f} | {(fp/1e9):12.3f} | {fr_pct:8.2f}%"
            )
        
        log.info("-" * 120)



    # ----------------------------
    # Optional benchmarking (NEW)
    # ----------------------------
    if args.bench:
        log.info("=" * 92)
        log.info("INFERENCE BENCHMARK (Optional)")
        log.info("=" * 92)

        bench_batch = int(args.bench_batch)
        bench_warmup = int(args.bench_warmup)
        bench_runs = int(args.bench_runs)
        bench_amp = bool(int(args.bench_amp) == 1)
        bench_cl = bool(int(args.bench_channels_last) == 1)
        bench_threads = int(args.bench_threads)

        # CPU benchmark
        net_cpu = UNet(n_channels=3, n_classes=2, f_channels=save_pruned_txt).cpu()
        net_cpu.load_state_dict(load_state_dict_weights_only(save_finetuned, map_location="cpu"))
        cpu_stats = benchmark_inference_avg_ms(
            net_cpu,
            device=torch.device("cpu"),
            input_hw=input_hw,
            batch_size=bench_batch,
            warmup=bench_warmup,
            runs=bench_runs,
            use_amp=False,
            channels_last=bench_cl,
            threads=bench_threads,
        )

        # GPU benchmark if available
        gpu_stats = None
        if torch.cuda.is_available():
            net_gpu = UNet(n_channels=3, n_classes=2, f_channels=save_pruned_txt).cuda()
            net_gpu.load_state_dict(load_state_dict_weights_only(save_finetuned, map_location="cuda"))
            gpu_stats = benchmark_inference_avg_ms(
                net_gpu,
                device=torch.device("cuda"),
                input_hw=input_hw,
                batch_size=bench_batch,
                warmup=bench_warmup,
                runs=bench_runs,
                use_amp=bench_amp,
                channels_last=bench_cl,
                threads=0,
            )

        # Report
        log.info(f"CPU avg={cpu_stats['avg_ms']:.3f} ms (std={cpu_stats['std_ms']:.3f}) | runs={bench_runs} warmup={bench_warmup}")
        if gpu_stats is not None:
            log.info(f"GPU avg={gpu_stats['avg_ms']:.3f} ms (std={gpu_stats['std_ms']:.3f}) | amp={int(bench_amp)}")
            if (cpu_stats['avg_ms'] > 0) and (gpu_stats['avg_ms'] > 0):
                speedup = float(cpu_stats['avg_ms']) / float(gpu_stats['avg_ms'])
                log.info(f"Speedup (CPU/GPU): {speedup:.2f}×")
        else:
            log.info("GPU benchmark skipped (CUDA not available).")


if __name__ == "__main__":
    main()
