import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import FGFNet
from tool.metrics import segmentation_metrics
from tool.whu_dataset import N_CLASSES, WHUTestDataset, WHUTrainDataset


def cross_entropy_2d(logits, target, weight=None):
    if logits.dim() != 4:
        raise ValueError(f"Expected logits with shape [N, C, H, W], got {tuple(logits.shape)}.")
    return F.cross_entropy(logits, target, weight=weight)


def evaluate(model, dataset, batch_size, device, num_workers):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    model.eval()
    all_preds = []
    pred = np.zeros(dataset.lbl_shape_list[0] + (N_CLASSES,), dtype=np.float32)
    image_idx_now = 0
    global_patch_idx = 0

    with torch.no_grad():
        for optical, sar in tqdm(loader, desc="Evaluate", leave=False):
            optical = optical.to(device, non_blocking=True)
            sar = sar.to(device, non_blocking=True)
            output = model(optical, sar).cpu().numpy()

            for i in range(output.shape[0]):
                idx, x, y = dataset.test_ixy[global_patch_idx]
                global_patch_idx += 1
                if idx != image_idx_now:
                    all_preds.append(np.argmax(pred, axis=-1))
                    image_idx_now = idx
                    pred = np.zeros(dataset.lbl_shape_list[image_idx_now] + (N_CLASSES,), dtype=np.float32)
                patch_w, patch_h = dataset.window_size
                pred[x:x + patch_w, y:y + patch_h] += output[i].transpose((1, 2, 0))

    all_preds.append(np.argmax(pred, axis=-1))
    foreground = [False, True, True, True, True, True, True, False]
    return segmentation_metrics(dataset.get_lbl(), all_preds, N_CLASSES, foreground=foreground)


def save_checkpoint(model, output_dir, run_name, epoch, score):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{run_name}_{epoch}_{score:.3f}.model"
    torch.save(model.state_dict(), path)
    return path


def main():
    parser = argparse.ArgumentParser(description="Train FGFNet V1 on the WHU OPT-SAR dataset.")
    parser.add_argument("--data-root", required=True, help="Path to WHU_OPT_SAR containing optical/, sar/, lbl/, and data.csv.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--iters-per-epoch", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--resume", default=None, help="Optional .model state_dict to load before training.")
    parser.add_argument("--output-dir", default="laboratory/bestModelsDir")
    parser.add_argument("--no-pretrained", action="store_true", help="Disable ImageNet initialization before training.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_iterations = args.epochs * args.iters_per_epoch
    train_dataset = WHUTrainDataset(
        args.data_root,
        set_len=args.batch_size * args.iters_per_epoch,
        window_size=(256, 256),
        augmentation=True,
    )
    test_dataset = WHUTestDataset(args.data_root, window_size=(256, 256), stride=args.stride)

    model = FGFNet(num_class=N_CLASSES, input_size=256, pretrained=not args.no_pretrained).to(device)
    if args.resume:
        state_dict = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iterations)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )

    run_name = f"train_FGFNet_WHU_{datetime.now().strftime('%Y%m%d')}_FGFNet"
    best_miou = 0.0
    best_path = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        for optical, sar, target in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            optical = optical.to(device, non_blocking=True)
            sar = sar.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            loss = cross_entropy_2d(model(optical, sar), target)
            loss.backward()
            optimizer.step()
            scheduler.step()

        should_eval = epoch == 1 or (epoch < 40 and epoch % 3 == 0) or (epoch >= 40 and epoch % 2 == 0)
        if should_eval:
            metrics = evaluate(model, test_dataset, args.batch_size, device, args.num_workers)
            miou = metrics["miou"]
            print(
                f"Epoch {epoch}: OA={metrics['OA']:.3f}, "
                f"mF1={metrics['mF1']:.4f}, Kappa={metrics['Kappa']:.4f}, miou={miou:.4f}"
            )
            if miou > best_miou:
                best_miou = miou
                best_path = save_checkpoint(model, Path(args.output_dir), run_name, epoch, miou)
                print(f"Saved best model: {best_path}")

    if best_path is None:
        print("Training finished without saving a checkpoint.")
    else:
        print(f"Training finished. Best miou={best_miou:.4f}, checkpoint={best_path}")


if __name__ == "__main__":
    main()
