import numpy as np
import torch
import torch.utils.data as data
import h5py


class DatasetFromHdf5(data.Dataset):

















    def __init__(self, file_path, gt_patch_size=64, scale=4, patches_per_scene=256):
        super(DatasetFromHdf5, self).__init__()
        self.file_path = file_path
        self.dataset = None
        self.gt_patch_size = int(gt_patch_size)
        self.scale = int(scale)
        self.lr_patch_size = self.gt_patch_size // self.scale
        self.patches_per_scene = int(patches_per_scene)

        self.mode = None
        self.index_entries = []
        self.GT = None
        self.LRHSI = None
        self.HRMSI = None

        self._open_dataset()
        self._init_layout()

    def _open_dataset(self):
        if self.dataset is None:
            self.dataset = h5py.File(self.file_path, "r")

    def _ensure_open(self):
        if self.dataset is None:
            self._open_dataset()

            self._init_layout()

    def __getstate__(self):

        state = self.__dict__.copy()
        state["dataset"] = None
        state["GT"] = None
        state["LRHSI"] = None
        state["HRMSI"] = None
        return state

    def __del__(self):
        try:
            if self.dataset is not None:
                self.dataset.close()
        except Exception:
            pass

    def _init_layout(self):
        self.index_entries = []
        gt_obj = self.dataset.get("GT")
        lr_obj = self.dataset.get("LRHSI")
        hr_obj = self.dataset.get("HRMSI") or self.dataset.get("RGB")

        if gt_obj is None or lr_obj is None or hr_obj is None:
            raise KeyError("HDF5 missing required keys: GT/LRHSI/HRMSI(or RGB)")


        if isinstance(gt_obj, h5py.Dataset) and isinstance(lr_obj, h5py.Dataset) and isinstance(hr_obj, h5py.Dataset):
            if gt_obj.ndim == 3 and lr_obj.ndim == 3 and hr_obj.ndim == 3:
                self.mode = "single"
                self.GT = gt_obj
                self.LRHSI = lr_obj
                self.HRMSI = hr_obj
                return
            if gt_obj.ndim != 4 or lr_obj.ndim != 4 or hr_obj.ndim != 4:
                raise ValueError(
                    "Flat datasets must all be 3D single scenes or 4D sample batches; "
                    f"got GT={gt_obj.shape}, LRHSI={lr_obj.shape}, HRMSI/RGB={hr_obj.shape}"
                )
            if gt_obj.shape[0] != lr_obj.shape[0] or gt_obj.shape[0] != hr_obj.shape[0]:
                raise ValueError("Flat datasets have mismatched sample counts")
            self.mode = "flat"
            self.GT = gt_obj
            self.LRHSI = lr_obj
            self.HRMSI = hr_obj
            return


        self.mode = "grouped"
        self._build_grouped_index()
        if len(self.index_entries) == 0:
            raise ValueError("No valid samples found in grouped HDF5 file")

    def _build_grouped_index(self):
        gt_root = self.dataset.get("GT")
        lr_root = self.dataset.get("LRHSI")
        hr_root = self.dataset.get("HRMSI") or self.dataset.get("RGB")

        if not isinstance(gt_root, h5py.Group) or not isinstance(lr_root, h5py.Group) or not isinstance(hr_root, h5py.Group):
            raise ValueError("Grouped layout expects GT/LRHSI/HRMSI as groups")

        gt_scenes = set(gt_root.keys())
        lr_scenes = set(lr_root.keys())
        hr_scenes = set(hr_root.keys())
        common_scenes = sorted(gt_scenes & lr_scenes & hr_scenes)

        for scene in common_scenes:
            gt_path = self._resolve_dataset_path("GT", scene, ["hyperspectral", "GT"])
            lr_path = self._resolve_dataset_path("LRHSI", scene, ["hyperspectral", "LRHSI"])
            hr_path = self._resolve_dataset_path("HRMSI", scene, ["HRMSI", "RGB"])

            if gt_path is None or lr_path is None or hr_path is None:
                continue

            gt_ds = self.dataset[gt_path]
            lr_ds = self.dataset[lr_path]
            hr_ds = self.dataset[hr_path]


            if gt_ds.ndim == 4 and lr_ds.ndim == 4 and hr_ds.ndim == 4:
                n = min(gt_ds.shape[0], lr_ds.shape[0], hr_ds.shape[0])
                for i in range(n):
                    self.index_entries.append({"kind": "stack", "gt": gt_path, "lr": lr_path, "hr": hr_path, "i": i})
                continue


            if gt_ds.ndim == 3 and lr_ds.ndim == 3 and hr_ds.ndim == 3:
                for _ in range(self.patches_per_scene):
                    self.index_entries.append({"kind": "scene", "gt": gt_path, "lr": lr_path, "hr": hr_path})

    def _resolve_dataset_path(self, root, scene, candidates):
        base = f"{root}/{scene}"
        for name in candidates:
            path = f"{base}/{name}"
            if path in self.dataset:
                return path
        return None

    def _to_chw(self, arr):
        if arr.ndim != 3:
            raise ValueError(f"Expect 3D array, got shape {arr.shape}")
        channel_sizes = (3, 31, 128)
        first_is_channel = arr.shape[0] in channel_sizes
        last_is_channel = arr.shape[-1] in channel_sizes




        is_hwc = last_is_channel and not first_is_channel
        if first_is_channel and last_is_channel:
            if arr.shape[0] == arr.shape[1] and arr.shape[1] != arr.shape[2]:
                is_hwc = True
            elif arr.shape[1] == arr.shape[2] and arr.shape[0] != arr.shape[1]:
                is_hwc = False

        if is_hwc:
            arr = np.transpose(arr, (2, 0, 1))
        return arr

    def _normalize(self, arr):

        if np.issubdtype(arr.dtype, np.integer):
            max_val = float(np.iinfo(arr.dtype).max)
            arr = arr.astype(np.float32) / max_val
        else:
            arr = arr.astype(np.float32)
        return arr

    def _random_crop_triplet(self, gt, lr, hr):

        _, h_gt, w_gt = gt.shape
        _, h_lr, w_lr = lr.shape

        if h_gt < self.gt_patch_size or w_gt < self.gt_patch_size:
            raise ValueError(f"GT image too small for patch size {self.gt_patch_size}: {(h_gt, w_gt)}")
        if h_lr < self.lr_patch_size or w_lr < self.lr_patch_size:
            raise ValueError(f"LRHSI image too small for patch size {self.lr_patch_size}: {(h_lr, w_lr)}")

        max_top_gt = h_gt - self.gt_patch_size
        max_left_gt = w_gt - self.gt_patch_size
        top_gt = np.random.randint(0, max_top_gt + 1)
        left_gt = np.random.randint(0, max_left_gt + 1)

        top_lr = top_gt // self.scale
        left_lr = left_gt // self.scale

        gt_patch = gt[:, top_gt:top_gt + self.gt_patch_size, left_gt:left_gt + self.gt_patch_size]
        hr_patch = hr[:, top_gt:top_gt + self.gt_patch_size, left_gt:left_gt + self.gt_patch_size]
        lr_patch = lr[:, top_lr:top_lr + self.lr_patch_size, left_lr:left_lr + self.lr_patch_size]

        return gt_patch, lr_patch, hr_patch

    def __getitem__(self, index):
        self._ensure_open()
        if self.mode == "single":
            if index != 0:
                raise IndexError(index)
            gt = self._normalize(self._to_chw(np.array(self.GT)))
            lr = self._normalize(self._to_chw(np.array(self.LRHSI)))
            hr = self._normalize(self._to_chw(np.array(self.HRMSI)))
            return torch.from_numpy(np.ascontiguousarray(gt)), torch.from_numpy(np.ascontiguousarray(lr)), torch.from_numpy(np.ascontiguousarray(hr))

        if self.mode == "flat":
            gt = self._normalize(self._to_chw(np.array(self.GT[index])))
            lr = self._normalize(self._to_chw(np.array(self.LRHSI[index])))
            hr = self._normalize(self._to_chw(np.array(self.HRMSI[index])))
            return torch.from_numpy(np.ascontiguousarray(gt)), torch.from_numpy(np.ascontiguousarray(lr)), torch.from_numpy(np.ascontiguousarray(hr))

        entry = self.index_entries[index]
        gt_arr = self._normalize(self._to_chw(np.array(self.dataset[entry["gt"]][entry.get("i", slice(None))])))
        lr_arr = self._normalize(self._to_chw(np.array(self.dataset[entry["lr"]][entry.get("i", slice(None))])))
        hr_arr = self._normalize(self._to_chw(np.array(self.dataset[entry["hr"]][entry.get("i", slice(None))])))

        if entry["kind"] == "scene":
            gt_arr, lr_arr, hr_arr = self._random_crop_triplet(gt_arr, lr_arr, hr_arr)

        return (
            torch.from_numpy(np.ascontiguousarray(gt_arr)),
            torch.from_numpy(np.ascontiguousarray(lr_arr)),
            torch.from_numpy(np.ascontiguousarray(hr_arr)),
        )

    def __len__(self):
        self._ensure_open()
        if self.mode == "single":
            return 1
        if self.mode == "flat":
            return int(self.GT.shape[0])
        return len(self.index_entries)
