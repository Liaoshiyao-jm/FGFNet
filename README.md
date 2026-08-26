# FGFNet

Clean PyTorch implementation of **Frequency-band Gated Fusion Network (FGFNet)** for WHU OPT-SAR semantic segmentation.

This repository provides a clean, runnable implementation extracted from the original experiment code and verified with the checkpoint:

`train_FGFNet_WHU_20260125_FGFNet_96_0.633.model`

The network path is fixed to the checkpoint-matched configuration:

- RGB optical input: 3 channels
- SAR input: 1 channel
- Backbone: ResNet-50 encoder for both modalities
- Decoder: convolution decoder
- Classes: 8 WHU classes
- Input patch size: 256 x 256
- Fusion: FADFEM with frequency dynamic spatial fusion and cross-layer frequency gating enabled

## Folder

```text
FGFNet/
  train.py
  model/
    encoder.py
    decoder.py
    fgfnet.py
  tool/
    whu_dataset.py
    metrics.py
```

## Dataset

Expected WHU dataset layout:

```text
WHU_OPT_SAR/
  data.csv
  train_ids.csv        # optional
  test_ids.csv         # optional
  optical/
    <id>.tif
  sar/
    <id>.tif
  lbl/
    <id>.tif
```

If `train_ids.csv` and `test_ids.csv` are not present, `data.csv` is split 90 percent for training and 10 percent for testing.

## Train

```bash
python train.py --data-root /path/to/WHU_OPT_SAR
```

Common options:

```bash
python train.py \
  --data-root /path/to/WHU_OPT_SAR \
  --epochs 50 \
  --iters-per-epoch 1000 \
  --batch-size 10 \
  --lr 1e-4 \
  --stride 64
```

Resume or fine-tune from the verified checkpoint:

```bash
python train.py \
  --data-root /path/to/WHU_OPT_SAR \
  --resume train_FGFNet_WHU_20260125_FGFNet_96_0.633.model
```

## Checkpoint Compatibility

The checkpoint above was verified with strict `state_dict` loading against this FGFNet implementation, and a random forward pass returns:

```text
(1, 8, 256, 256)
```
