import torch
from unet import UNet

# ---- PART 1: print checkpoint keys ----
print("\n=== Checkpoint keys ===\n")
ck = torch.load("unet_carvana_scale0.5_epoch2.pth", map_location="cpu")

# if wrapped
if "state_dict" in ck:
    ck = ck["state_dict"]

for k, v in ck.items():
    print(k, tuple(v.shape))


# ---- PART 2: print current model keys ----
print("\n=== Current model keys ===\n")

net = UNet(
    n_channels=3,
    n_classes=1,
    f_channels="model_channels.txt"
)

for k, v in net.state_dict().items():
    print(k, tuple(v.shape))
