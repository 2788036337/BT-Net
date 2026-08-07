import argparse
import time

import h5py
import numpy as np
import torch

from model import build_model


DATASETS = [
    ("test_dataset", "datasets/test_dataset.h5"),
    ("chikusei", "chikusei_test_6areas.h5"),
    ("harvard", "harvard_dataset_10.h5"),
]


def _to_float_tensor(arr, device):
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.float32) / float(np.iinfo(arr.dtype).max)
    else:
        arr = arr.astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(arr)).to(device=device, non_blocking=False)


def _open_flat_scene(file_handle, idx, device):
    gt = _to_float_tensor(file_handle["GT"][idx], device)
    lrhsi = _to_float_tensor(file_handle["LRHSI"][idx], device)
    hrmsi_key = "HRMSI" if "HRMSI" in file_handle else "RGB"
    hrmsi = _to_float_tensor(file_handle[hrmsi_key][idx], device)
    return gt, lrhsi, hrmsi


def _warmup(model, device, steps, gt_patch_size, scale, hsi_channels, msi_channels):
    lr_patch_size = gt_patch_size // scale
    lrhsi = torch.zeros((1, hsi_channels, lr_patch_size, lr_patch_size), device=device)
    hrmsi = torch.zeros((1, msi_channels, gt_patch_size, gt_patch_size), device=device)
    with torch.inference_mode():
        for _ in range(steps):
            model(lrhsi, hrmsi)
    torch.cuda.synchronize()


def measure_dataset(model, dataset_path, device, gt_patch_size=64, scale=4):
    lr_patch_size = gt_patch_size // scale
    io_seconds = 0.0
    wall_seconds = 0.0
    gpu_seconds = 0.0
    scene_count = 0
    patch_count = 0

    with h5py.File(dataset_path, "r") as file_handle:
        scene_count = int(file_handle["GT"].shape[0])
        for idx in range(scene_count):
            io_start = time.perf_counter()
            gt, lrhsi, hrmsi = _open_flat_scene(file_handle, idx, device)
            torch.cuda.synchronize()
            io_seconds += time.perf_counter() - io_start

            _, h_gt, w_gt = gt.shape
            out_full = torch.empty_like(gt)

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            wall_start = time.perf_counter()
            start_event.record()
            with torch.inference_mode():
                for top in range(0, h_gt, gt_patch_size):
                    for left in range(0, w_gt, gt_patch_size):
                        lr_top = top // scale
                        lr_left = left // scale
                        lr_patch = lrhsi[
                            :,
                            lr_top:lr_top + lr_patch_size,
                            lr_left:lr_left + lr_patch_size,
                        ].unsqueeze(0)
                        hr_patch = hrmsi[
                            :,
                            top:top + gt_patch_size,
                            left:left + gt_patch_size,
                        ].unsqueeze(0)
                        output_patch, _, _ = model(lr_patch, hr_patch)
                        output_patch = output_patch.clamp(0, 1)
                        out_full[
                            :,
                            top:top + gt_patch_size,
                            left:left + gt_patch_size,
                        ] = output_patch[0]
                        patch_count += 1
            end_event.record()
            torch.cuda.synchronize()
            wall_seconds += time.perf_counter() - wall_start
            gpu_seconds += start_event.elapsed_time(end_event) / 1000.0

            del gt, lrhsi, hrmsi, out_full

    return {
        "scenes": scene_count,
        "patches": patch_count,
        "gpu_seconds": gpu_seconds,
        "wall_seconds": wall_seconds,
        "io_seconds": io_seconds,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="bt-net", choices=["bt-net"])
    parser.add_argument("--ckpt", default="Trained_model/300.pth")
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to measure GPU inference time.")

    device = torch.device("cuda")
    checkpoint = torch.load(args.ckpt, map_location=device)
    state_dict = checkpoint.get("net", checkpoint)
    if next(iter(state_dict)).startswith("module."):
        state_dict = {key[7:]: value for key, value in state_dict.items()}
    hsi_channels = int(state_dict["refine.2.weight"].shape[0])
    input_channels = int(state_dict["embedding.weight"].shape[1])
    msi_channels = input_channels - hsi_channels
    model = build_model(args.model_name, num_channel=hsi_channels,
                        msi_channels=msi_channels).to(device).eval()
    model.load_state_dict(state_dict, strict=True)

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Model: {args.model_name}")
    print(f"Checkpoint: {args.ckpt}")
    print("Patch: GT 64x64, LRHSI 16x16, scale=4")

    _warmup(model, device, args.warmup, gt_patch_size=64, scale=4,
            hsi_channels=hsi_channels, msi_channels=msi_channels)

    for name, path in DATASETS:
        stats = measure_dataset(model, path, device)
        scenes = stats["scenes"]
        patches = stats["patches"]
        print(
            f"{name}: scenes={scenes}, patches={patches}, "
            f"GPU_forward={stats['gpu_seconds']:.4f}s, "
            f"wall_infer={stats['wall_seconds']:.4f}s, "
            f"io_h2d={stats['io_seconds']:.4f}s, "
            f"GPU/scene={stats['gpu_seconds'] / scenes:.4f}s, "
            f"wall/scene={stats['wall_seconds'] / scenes:.4f}s, "
            f"GPU/patch={stats['gpu_seconds'] / patches * 1000:.4f}ms"
        )


if __name__ == "__main__":
    main()
