# generate_model_channels.py
import torch
import torch.nn as nn
from unet import UNet

CKPT = "unet_carvana_scale0.5_epoch2.pth"
OUT_TXT = "model_channels_from_ckpt.txt"

# 1) load checkpoint (handle wrapped)
ck = torch.load(CKPT, map_location="cpu")
if "state_dict" in ck and isinstance(ck["state_dict"], dict):
    ck = ck["state_dict"]

# 2) derive n_classes from checkpoint final conv if possible
# try common final-layer keys in checkpoint
possible_out_keys = ["outc.conv.weight", "outc.weight", "out.weight", "outc.conv.0.weight"]
n_classes_ck = None
for k in ck.keys():
    if k.endswith("outc.conv.weight") or k.endswith("outc.weight") or k.endswith("out.weight"):
        n_classes_ck = ck[k].shape[0]
        break

if n_classes_ck is None:
    # fallback: inspect any key that looks like final conv in your printed ckpt
    if "outc.conv.weight" in ck:
        n_classes_ck = ck["outc.conv.weight"].shape[0]
    else:
        n_classes_ck = 2  # safe default if you inspected earlier

print("Detected checkpoint n_classes (outputs):", n_classes_ck)

# 3) build a net with same number of classes so shapes match for loading
net = UNet(n_channels=3, n_classes=n_classes_ck, f_channels="model_channels.txt")  # f_channels argument won't be used here

# 4) rename checkpoint keys heuristically (same rules we used earlier)
def rename_key(k):
    k2 = k
    k2 = k2.replace("double_conv", "conv.conv")
    k2 = k2.replace("maxpool_conv", "mpconv")
    # add other renames if needed
    return k2

renamed_ck = {rename_key(k): v for k, v in ck.items()}

# 5) try to load renamed_ck into model (copy only matching shapes)
model_state = net.state_dict()
copied = []
skipped = []
for my_k in list(model_state.keys()):
    if my_k in renamed_ck:
        v = renamed_ck[my_k]
        if tuple(v.shape) == tuple(model_state[my_k].shape):
            model_state[my_k] = v.clone()
            copied.append(my_k)
        else:
            skipped.append((my_k, tuple(v.shape), tuple(model_state[my_k].shape)))
    else:
        skipped.append((my_k, None, tuple(model_state[my_k].shape)))

# Load the partial state (this will keep random init for skipped params)
net.load_state_dict(model_state)

print(f"Copied {len(copied)} parameters from checkpoint (exact-shape matches).")
if skipped:
    print(f"Skipped {len(skipped)} params due to missing key or shape mismatch. Example skips:")
    for s in skipped[:8]:
        print(" ", s)

# 6) collect conv modules in the same order Pruner would (named_modules order)
conv_modules = []
for name, module in net.named_modules():
    if isinstance(module, nn.Conv2d):
        # skip final outc? Pruner's channel_save used self.convs[1:-1] so it expects conv list including first and last
        conv_modules.append((name, module))

print("Collected conv modules (in order):")
for n, m in conv_modules:
    print(" ", n, tuple(m.weight.shape))

# 7) reproduce Pruner.channel_save() logic
chans = []
# In original pruner.channel_save:
# for i, c in enumerate(self.convs[1:-1]):
#     if (i > 8 and (i-9) % 2 == 0) or i == 0:
#         chans.append(c.weight.shape[1])
#     chans.append(c.weight.shape[0])
#
# Note: that function used self.convs[1:-1] (skipping first and last conv in convs list)
# We'll replicate that: build convs list, then take convs[1:-1]
conv_only = [c for (_, c) in conv_modules]
if len(conv_only) < 3:
    raise RuntimeError("Unexpected number of conv modules found in the model.")

for i, c in enumerate(conv_only[1:-1]):
    # the 'i' here is relative to conv_only[1:-1], same as Pruner
    if (i > 8 and (i - 9) % 2 == 0) or i == 0:
        chans.append(c.weight.shape[1])  # input channels for these selected entries
    chans.append(c.weight.shape[0])      # output channels for each

# 8) write to file
with open(OUT_TXT, "w") as f:
    for item in chans:
        f.write(f"{item}\n")

print(f"Wrote {len(chans)} channel numbers to {OUT_TXT}. Sample:")
print(chans[:40])
