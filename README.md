# BT-Net

BT-Net is a hybrid Transformer and Mamba-like network for hyperspectral image
super-resolution (HSI-SR).

## Repository contents

- `main.py`: training, validation, and testing entry point.
- `model.py`: BT-Net and its network components.
- `data.py`: HDF5 dataset loader.
- `eval_metrics.py`: standalone evaluation with PSNR, SAM, CC, ERGAS, and SSIM.
- `measure_inference_time.py`: inference-time measurement.
- `datasets/split_config.py`: local train/validation/test dataset paths.

Datasets, training logs, and generated results are intentionally excluded from
Git. The repository includes the epoch-200 BT-Net checkpoint at
`checkpoints/bt_net_epoch_200.pth`. Put local HDF5 files under `datasets/` or
update `datasets/split_config.py`.

## Dataset downloads

- [CAVE multispectral image database](https://cave.cs.columbia.edu/repository/Multispectral/)
- [Harvard hyperspectral image database](https://vision.seas.harvard.edu/hyperspec/)
- [Chikusei hyperspectral dataset](http://park.itc.u-tokyo.ac.jp/sal/hyperdata/)

## Installation

```bash
pip install -r requirements.txt
```

## Data layout

The loader supports flat HDF5 tensors and grouped scene HDF5 files. Each sample
must provide:

- `GT`: high-resolution hyperspectral ground truth.
- `LRHSI`: low-resolution hyperspectral input.
- `HRMSI` or `RGB`: high-resolution multispectral input.

The default paths are:

```text
datasets/train_dataset.h5
datasets/val_dataset.h5
datasets/test_dataset.h5
```

## Training and testing

The current experiment switches are near the bottom of `main.py`:

```python
train_or_not = 1
test_or_not = 0
resume_or_not = 0
```

Run the experiment with:

```bash
python main.py
```

## Evaluation metrics

Evaluate a checkpoint on the validation set:

```bash
python eval_metrics.py --ckpt checkpoints/bt_net_epoch_200.pth \
  --val datasets/val_dataset.h5
```

The script reports:

- PSNR (dB; higher is better)
- SAM (degrees; lower is better)
- CC (higher is better)
- ERGAS (lower is better)
- SSIM (higher is better)

The reusable metric functions are `psnr_batch`, `sam_batch`, `cc_batch`,
`ergas_batch`, and `ssim_batch` in `eval_metrics.py`.

## Inference time

```bash
python measure_inference_time.py --help
```

## License

This project is licensed under the [MIT License](LICENSE).
