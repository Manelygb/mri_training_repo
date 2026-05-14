import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from dice_loss import dice_coeff
from utils import batch


def _format_true_masks(true_masks: torch.Tensor) -> torch.Tensor:
    """
    Returns target mask as LongTensor [B,H,W] with values in {0,1}.
    Accepts:
      - [B,H,W]
      - [B,1,H,W]
      - {0,255} masks
    """
    if true_masks.ndim == 4 and true_masks.shape[1] == 1:
        true_masks = true_masks[:, 0, :, :]  # [B,H,W]

    # binarize if 0/255
    if true_masks.max() > 1:
        true_masks = (true_masks > 0).long()
    else:
        true_masks = true_masks.long()

    return true_masks


def eval_net(net, dataset, lendata, gpu=False, batch_size=8, is_loss=False):
    """
    Evaluation with Dice (default) or loss (is_loss=True).

    Supports:
      - 2-class logits output: [B,2,H,W]  (CrossEntropy-style)  -> uses argmax
      - 1-channel output:      [B,1,H,W]  (sigmoid-style)       -> uses sigmoid+threshold for dice
    """
    was_training = net.training
    net.eval()

    device = torch.device("cuda" if (gpu and torch.cuda.is_available()) else "cpu")
    criterion_bce = nn.BCELoss()

    tot = 0.0
    n_batches = 0

    with torch.no_grad(), tqdm(total=lendata) as progress_bar:
        for b in batch(dataset, batch_size):
            imgs = np.array([i[0] for i in b]).astype(np.float32)     # [B,3,H,W]
            true_masks = np.array([i[1] for i in b])                  # [B,1,H,W] or [B,H,W]

            imgs = torch.from_numpy(imgs).to(device=device, dtype=torch.float32)
            true_masks = torch.from_numpy(true_masks).to(device=device)

            true_masks = _format_true_masks(true_masks)               # [B,H,W] long 0/1

            out = net(imgs)                                            # could be [B,2,H,W] or [B,1,H,W] or tuple/list

            # Some codebases return (logits,) or (logits, aux...). Handle that safely:
            if isinstance(out, (tuple, list)):
                out = out[0]

            if is_loss:
                # Loss mode: only defined cleanly for 1-channel sigmoid models.
                # If you want CE loss for 2-channel logits, switch to CrossEntropyLoss here.
                if out.ndim == 4 and out.shape[1] == 2:
                    ce = nn.CrossEntropyLoss()
                    loss = ce(out, true_masks)  # out [B,2,H,W], target [B,H,W]
                    tot += float(loss.item())
                    progress_bar.set_postfix(CE=float(loss.item()))
                else:
                    # 1-channel: apply sigmoid then BCE on float masks
                    if out.ndim == 4 and out.shape[1] == 1:
                        out = out[:, 0, :, :]  # [B,H,W]
                    prob = torch.sigmoid(out)  # [B,H,W]
                    loss = criterion_bce(prob, true_masks.float())
                    tot += float(loss.item())
                    progress_bar.set_postfix(BCE=float(loss.item()))
            else:
                # Dice mode
                if out.ndim == 4 and out.shape[1] == 2:
                    # 2-class logits -> predicted class mask
                    pred = torch.argmax(out, dim=1)                    # [B,H,W] 0/1
                else:
                    # 1-channel -> sigmoid + threshold
                    if out.ndim == 4 and out.shape[1] == 1:
                        out = out[:, 0, :, :]                          # [B,H,W]
                    prob = torch.sigmoid(out)
                    pred = (prob > 0.5).long()                         # [B,H,W]

                # dice_coeff usually expects float tensors
                dice = dice_coeff(pred.float(), true_masks.float()).item()
                tot += float(dice)
                progress_bar.set_postfix(DICE=float(dice))

            n_batches += 1
            progress_bar.update(len(b))  # update by actual batch length (handles last batch)
            
    result = tot / max(1, n_batches)
    if was_training:
        net.train()
    return result
    # return tot / max(1, n_batches)
