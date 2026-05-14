# finetune.py (replacement)
import os
import os.path as osp

import torch
import numpy as np
from tqdm import tqdm

from utils import batch, AverageMeter, get_imgs_and_masks
from flops_counter import flops_count

def _ensure_trailing_sep(p: str) -> str:
    p = osp.expanduser(p)
    return p if p.endswith(os.sep) else p + os.sep

def _resolve_data_dirs() -> tuple[str, str]:
    return (
        _ensure_trailing_sep("dataset_split/train/imgs"),
        _ensure_trailing_sep("dataset_split/train/masks"),
    )

def finetune(net, optimizer, criterion, trainset, log, path,
             iters=100, epochs=None, batch_size=2, gpu=True, scale=0.5):
    net.train()
    loss_meter = AverageMeter()

    dir_img, dir_mask = _resolve_data_dirs()
    log.info(f"[finetune] Using dir_img={dir_img} dir_mask={dir_mask}")

    if epochs is None:  # Fine-tune using iterations of mini-batches
        epochs = 1
    else:               # Fine-tune using entire epochs
        iters = None

    for e in range(epochs):
        # reset the generators
        train = get_imgs_and_masks(trainset, dir_img, dir_mask, scale)

        with tqdm(total=len(trainset)) as progress_bar:
            for i, b in enumerate(batch(train, batch_size)):
                imgs = np.array([i0[0] for i0 in b]).astype(np.float32)
                true_masks = np.array([i0[1] for i0 in b])

                imgs = torch.from_numpy(imgs)                  # [B,3,H,W] float32
                true_masks = torch.from_numpy(true_masks)      # could be [B,1,H,W] or [B,H,W]

                # ---- Target formatting for CrossEntropyLoss ----
                # Need: target = [B,H,W] of dtype long with values in {0,1}
                if true_masks.ndim == 4 and true_masks.shape[1] == 1:
                    true_masks = true_masks[:, 0, :, :]        # [B,H,W]

                # if masks are 0/255, binarize to 0/1
                if true_masks.max() > 1:
                    true_masks = (true_masks > 0).long()
                else:
                    true_masks = true_masks.long()

                if gpu:
                    imgs = imgs.cuda(non_blocking=True)
                    true_masks = true_masks.cuda(non_blocking=True)

                # ---- Forward: keep logits as float, NO softmax, NO argmax ----
                masks_pred = net(imgs)                         # [B,2,H,W] float logits

                # Safety: ensure float dtype for CE
                if not masks_pred.is_floating_point():
                    masks_pred = masks_pred.float()

                loss = criterion(masks_pred, true_masks)
                loss_meter.update(loss.item(), imgs.shape[0])

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                progress_bar.update(imgs.shape[0])
                progress_bar.set_postfix(epoch=e, loss=loss_meter.avg)

                if i == 0 and e == 0:
                    try:
                        log.info("FLOPs after pruning: \n{}".format(flops_count(net, imgs.shape[2:])))
                    except Exception as ex:
                        log.info(f"[finetune] FLOPs count failed: {ex}")

                if iters is not None and i >= iters:  # Stop finetuning after sufficient mini-batches
                    break

    log.info("Finished finetuning")
    log.info("Finetuned loss: {}".format(loss_meter.avg))
    torch.save(net.state_dict(), path)
    log.info("Saving finetuned to {}...".format(path))
