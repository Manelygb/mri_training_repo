import torch
import torch.nn as nn


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


    def _part_indices(self):
        """
        Returns conv indices per UNet part.
        """
        enc, bottleneck, dec = [], [], []

        for i, name in enumerate(self.conv_names):
            if name.startswith("inc.") or name.startswith("down"):
                if "down4" in name:
                    bottleneck.append(i)
                else:
                    enc.append(i)
            elif name.startswith("up") or name.startswith("outc"):
                dec.append(i)

        return enc, bottleneck, dec

    def prune_by_parts(self, total_channels: int, ratios: dict):
        """
        Dependency-aware pruning by ratios.
        Ratios are converted to channel counts internally.
        """

        assert abs(sum(ratios.values()) - 1.0) < 1e-6, \
            f"Ratios must sum to 1.0, got {ratios}"

        enc_ratio = ratios.get("encoder", 0.0)
        bn_ratio  = ratios.get("bottleneck", 0.0)
        dec_ratio = ratios.get("decoder", 0.0)

        enc_budget = int(round(total_channels * enc_ratio))
        bn_budget  = int(round(total_channels * bn_ratio))
        dec_budget = total_channels - enc_budget - bn_budget

        print(
            f"[Pruner] Budget split | "
            f"ENC={enc_budget}, BN={bn_budget}, DEC={dec_budget} "
            f"(Total={total_channels})"
        )

        applied_total = 0

        # -------- Encoder --------
        before = self._count_remaining_channels()
        self.pruning(enc_budget)
        after = self._count_remaining_channels()
        enc_applied = before - after
        applied_total += enc_applied

        # -------- Bottleneck --------
        remaining = total_channels - applied_total
        bn_budget = min(bn_budget, remaining)

        before = self._count_remaining_channels()
        self.pruning(bn_budget)
        after = self._count_remaining_channels()
        bn_applied = before - after
        applied_total += bn_applied

        # -------- Decoder --------
        remaining = total_channels - applied_total
        if remaining > 0:
            self.pruning(remaining)
            applied_total = total_channels

        print(
            f"[Pruner] Applied pruning | "
            f"ENC={enc_applied}, BN={bn_applied}, TOTAL={applied_total}"
        )

    def _count_remaining_channels(self):
        """
        Counts total remaining Conv2d output channels.
        """
        total = 0
        for c in self.convs:
            total += c.weight.shape[0]
        return total



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
