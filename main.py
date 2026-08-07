# Training and testing code for BT-Net.

# 导入必要的库
import os  # 文件和目录操作
# Windows下部分科学计算库会重复加载OpenMP运行时，这里允许进程继续执行。
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import multiprocessing as mp
import torch  # PyTorch深度学习框架
import torch.backends.cudnn as cudnn  # CUDA加速
import torch.nn as nn  # 神经网络模块
import torch.optim as optim  # 优化器
from torch.autograd import Variable  # 自动求导变量（较旧版本使用）
from torch.utils.data import DataLoader  # 数据加载器
from data import DatasetFromHdf5  # 自定义数据集类
# from model import PanNet,summaries  # 注释掉的其他模型
from model import *  # 导入模型相关类
import numpy as np  # 数值计算
import scipy.io as sio  # MATLAB文件读写
import shutil  # 文件操作
from torch.utils.tensorboard import SummaryWriter  # TensorBoard日志
import time  # 时间测量

try:
    from datasets.split_config import TRAIN_DATASET, VAL_DATASET, TEST_DATASET
except Exception:
    TRAIN_DATASET = "datasets/train_dataset.h5"
    VAL_DATASET = "datasets/val_dataset.h5"
    TEST_DATASET = "datasets/test_dataset.h5"

# ================== 预定义设置 =================== #
# 设置随机种子以确保结果可重现
SEED = 1
torch.manual_seed(SEED) # 设置CPU随机种子
torch.cuda.manual_seed(SEED) # 设置GPU随机种子
torch.cuda.manual_seed_all(SEED) # 设置所有GPU随机种子
cudnn.benchmark = True  # 启用cuDNN的自动优化
cudnn.deterministic = True  # 确保确定性结果

# ============= 超参数设置 ==========#
# 设置CUDA设备
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

# 学习率、训练轮数、检查点保存间隔、批大小
lr = 2e-4
epochs = 500
ckpt_step = 5
metric_step = 5
# 8GB左右显存下2通常会OOM，默认改为1更稳。
batch_size = 1
num_workers = min(8, max(2, (os.cpu_count() or 4) // 2))
use_amp = False
model_name = "bt-net"
num_hsi_channels = 31
num_msi_channels = 3
scale_factor = 4

# Windows下DataLoader共享内存映射容易触发1455，默认改为更稳配置。
IS_WINDOWS = (os.name == "nt")
if IS_WINDOWS:
    num_workers = 0

# 初始化模型并移到GPU
model = None
scaler = None
PLoss = None
optimizer = None
lr_scheduler = None

# TensorBoard日志目录
model_folder = "Trained_model/"
writer = None


def _psnr_batch(pred, gt, data_range=1.0, eps=1e-12):
    mse = torch.mean((pred - gt) ** 2, dim=(1, 2, 3))
    return 10.0 * torch.log10((data_range * data_range) / torch.clamp(mse, min=eps))


def _sam_batch(pred, gt, eps=1e-12):
    dot = torch.sum(pred * gt, dim=1)
    pred_norm = torch.linalg.norm(pred, dim=1)
    gt_norm = torch.linalg.norm(gt, dim=1)
    cos = dot / torch.clamp(pred_norm * gt_norm, min=eps)
    cos = torch.clamp(cos, -1.0, 1.0)
    angle = torch.acos(cos)
    angle_deg = angle * (180.0 / np.pi)
    return torch.mean(angle_deg, dim=(1, 2))


def _cc_batch(pred, gt, eps=1e-12):
    b, c, h, w = pred.shape
    pred_f = pred.reshape(b, c, h * w)
    gt_f = gt.reshape(b, c, h * w)

    pred_centered = pred_f - pred_f.mean(dim=2, keepdim=True)
    gt_centered = gt_f - gt_f.mean(dim=2, keepdim=True)

    numerator = torch.sum(pred_centered * gt_centered, dim=2)
    denominator = torch.sqrt(
        torch.sum(pred_centered ** 2, dim=2) * torch.sum(gt_centered ** 2, dim=2)
    )
    corr = numerator / torch.clamp(denominator, min=eps)
    corr = torch.clamp(corr, -1.0, 1.0)
    return torch.mean(corr, dim=1)


def _ergas_batch(pred, gt, scale_factor=4.0, eps=1e-12):
    rmse_band = torch.sqrt(torch.mean((pred - gt) ** 2, dim=(2, 3)))
    mean_gt_band = torch.mean(gt, dim=(2, 3))
    rel = rmse_band / torch.clamp(mean_gt_band, min=eps)
    ergas = (100.0 / float(scale_factor)) * torch.sqrt(torch.mean(rel ** 2, dim=1))
    return ergas


def _gaussian_window(window_size=11, sigma=1.5, device=None, dtype=None):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = g / g.sum()
    return torch.outer(g, g)


def _ssim_batch(pred, gt, data_range=1.0, window_size=11, sigma=1.5, eps=1e-12):
    b, c, _, _ = pred.shape
    window_2d = _gaussian_window(
        window_size=window_size,
        sigma=sigma,
        device=pred.device,
        dtype=pred.dtype,
    )
    kernel = window_2d.view(1, 1, window_size, window_size).repeat(c, 1, 1, 1)

    mu_x = torch.nn.functional.conv2d(pred, kernel, padding=window_size // 2, groups=c)
    mu_y = torch.nn.functional.conv2d(gt, kernel, padding=window_size // 2, groups=c)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = torch.nn.functional.conv2d(pred * pred, kernel, padding=window_size // 2, groups=c) - mu_x2
    sigma_y2 = torch.nn.functional.conv2d(gt * gt, kernel, padding=window_size // 2, groups=c) - mu_y2
    sigma_xy = torch.nn.functional.conv2d(pred * gt, kernel, padding=window_size // 2, groups=c) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    num = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim_map = num / torch.clamp(den, min=eps)
    return ssim_map.reshape(b, c, -1).mean(dim=(1, 2))


def init_training_state():
    """Initialize training-only globals in main process.

    On Windows, DataLoader workers use spawn and re-import this module.
    Keeping CUDA/model setup out of module import avoids worker bootstrap failures.
    """
    global model, scaler, PLoss, optimizer, lr_scheduler, writer

    model = build_model(model_name, num_channel=num_hsi_channels,
                        msi_channels=num_msi_channels,
                        scale_factor=scale_factor).cuda()
    # model = nn.DataParallel(model)  # 可选：多GPU并行
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    PLoss = nn.L1Loss(reduction='mean').cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=200, gamma=0.1)
    writer = SummaryWriter("train_logs/" + model_folder)
# 保存模型检查点的函数
def save_checkpoint(model, epoch):
    model_out_path = model_folder + "{}.pth".format(epoch)
    checkpoint = {
        "net": model.state_dict(),  # 模型权重
        'optimizer': optimizer.state_dict(),  # 优化器状态
        "epoch": epoch,  # 当前轮数
        "lr": lr  # 学习率
    }
    if not os.path.isdir(model_folder):
        os.mkdir(model_folder)  # 创建目录
    torch.save(checkpoint, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


###################################################################
# ------------------- 训练函数 ----------------------------------
###################################################################

def train(training_data_loader, validate_data_loader, start_epoch=0, RESUME=False, resume_ckpt=None):
    print('Start training...')

    if RESUME:
        path_checkpoint = resume_ckpt or (model_folder + "{}.pth".format(500))
        if not os.path.isfile(path_checkpoint):
            print(f"[Resume] Checkpoint not found: {path_checkpoint}. Start from scratch.")
        else:
            checkpoint = torch.load(path_checkpoint, map_location='cpu')
            state_dict = checkpoint.get('net', checkpoint)
            try:
                model.load_state_dict(state_dict, strict=True)
                if 'optimizer' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                start_epoch = int(checkpoint.get('epoch', 0))
                print('Network is Successfully Loaded from %s' % (path_checkpoint))
            except RuntimeError as e:
                print("[Resume] Model structure mismatch, skip loading this checkpoint.")
                print(f"[Resume] Detail: {e}")
                print("[Resume] Continue training from epoch 0 with current model initialization.")
    for epoch in range(start_epoch, epochs, 1):

        epoch += 1
        epoch_train_loss, epoch_val_loss = [], []
        epoch_time_s = time.time()
        data_time_total, compute_time_total = 0.0, 0.0

        # ============Epoch Train=============== #
        model.train()
        iter_end_t = time.time()

        for iteration, batch in enumerate(training_data_loader, 1):
            iter_start_t = time.time()
            data_time_total += (iter_start_t - iter_end_t)
            GT = batch[0].cuda(non_blocking=True)
            LRHSI = batch[1].cuda(non_blocking=True)
            HRMSI = batch[2].cuda(non_blocking=True)

            optimizer.zero_grad(set_to_none=True)  # set_to_none可减少显存写入开销

            if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                autocast_ctx = torch.amp.autocast(device_type='cuda', enabled=use_amp)
            else:
                autocast_ctx = torch.cuda.amp.autocast(enabled=use_amp)
            with autocast_ctx:
                output_HRHSI,UP_LRHSI,Highpass = model(LRHSI,HRMSI)
                Pixelwise_Loss =PLoss(output_HRHSI, GT)
            # Pixelwise_Loss  = PLoss(Rec_LRHSI, LRHSI) + PLoss(Rec_HRMSI,HRMSI)
            # Sparse = Sparse_loss(A) + Sparse_loss(A_LR)
            # ASC_loss = Sum2OneLoss(A) + Sum2OneLoss(A_LR)

            Myloss = Pixelwise_Loss
            if not torch.isfinite(Myloss):
                print(f"[Warn] Non-finite loss detected at epoch {epoch}, iter {iteration}. Skip this step.")
                optimizer.zero_grad(set_to_none=True)
                iter_end_t = time.time()
                continue
            epoch_train_loss.append(Myloss.item())  # save all losses into a vector for one epoch

            scaler.scale(Myloss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            compute_time_total += (time.time() - iter_start_t)
            iter_end_t = time.time()

            if iteration % 10 == 0:
                # log_value('Loss', loss.data[0], iteration + (epoch - 1) * len(training_data_loader))
                print("===> Epoch[{}]({}/{}): Loss: {:.6f}".format(epoch, iteration, len(training_data_loader),
                                                                   Myloss.item()))
                # for name, parameters in model.named_parameters():
                #     print(name, ':', parameters.size(), parameters)
            # for name, layer in model.named_parameters():
            #     # writer.add_histogram('torch/'+name + '_grad_weight_decay', layer.grad, epoch*iteration)
            #     writer.add_histogram('net/'+name + '_data_weight_decay', layer, epoch*iteration)
        print("learning rate:º%f" % (optimizer.param_groups[0]['lr']))
        lr_scheduler.step()  # update lr

        t_loss = np.nanmean(np.array(epoch_train_loss))  # compute the mean value of all losses, as one epoch loss
        epoch_time_e = time.time()
        epoch_elapsed = epoch_time_e - epoch_time_s
        train_iters = len(training_data_loader)
        print("Epoch timing | total: {:.2f}s | data: {:.4f}s/iter | compute: {:.4f}s/iter | it/s: {:.2f}".format(
            epoch_elapsed,
            data_time_total / max(1, train_iters),
            compute_time_total / max(1, train_iters),
            train_iters / max(1e-6, epoch_elapsed)
        ))
        writer.add_scalar('mse_loss/t_loss', t_loss, epoch)  # write to tensorboard to check
        print('Epoch: {}/{} training loss: {:.7f}'.format(epochs, epoch, t_loss))  # print loss for each epoch
        if epoch % ckpt_step == 0:  # if each ckpt epochs, then start to save model
            save_checkpoint(model, epoch)
        # ============Epoch Validate=============== #
        if epoch % metric_step == 0:
            model.eval()
            sam_vals, cc_vals, psnr_vals = [], [], []
            ergas_vals, ssim_vals = [], []
            with torch.no_grad():
                for iteration, batch in enumerate(validate_data_loader, 1):
                    GT, LRHSI, HRMSI = batch[0].cuda(), batch[1].cuda(), batch[2].cuda()
                    output_HRHSI, UP_LRHSI, Highpass = model(LRHSI, HRMSI)
                    metric_output = output_HRHSI.clamp(0, 1)
                    Pixelwise_Loss = PLoss(metric_output, GT)
                    epoch_val_loss.append(Pixelwise_Loss.item())

                    sam_vals.append(_sam_batch(metric_output, GT).detach().cpu().numpy())
                    cc_vals.append(_cc_batch(metric_output, GT).detach().cpu().numpy())
                    psnr_vals.append(_psnr_batch(metric_output, GT).detach().cpu().numpy())
                    ergas_vals.append(_ergas_batch(metric_output, GT).detach().cpu().numpy())
                    ssim_vals.append(_ssim_batch(metric_output, GT).detach().cpu().numpy())

            v_loss = np.nanmean(np.array(epoch_val_loss))
            sam_mean = float(np.concatenate(sam_vals).mean()) if sam_vals else float('nan')
            cc_mean = float(np.concatenate(cc_vals).mean()) if cc_vals else float('nan')
            psnr_mean = float(np.concatenate(psnr_vals).mean()) if psnr_vals else float('nan')
            ergas_mean = float(np.concatenate(ergas_vals).mean()) if ergas_vals else float('nan')
            ssim_mean = float(np.concatenate(ssim_vals).mean()) if ssim_vals else float('nan')

            writer.add_scalar('val/loss', v_loss, epoch)
            writer.add_scalar('val/SAM', sam_mean, epoch)
            writer.add_scalar('val/CC', cc_mean, epoch)
            writer.add_scalar('val/PSNR', psnr_mean, epoch)
            writer.add_scalar('val/ERGAS', ergas_mean, epoch)
            writer.add_scalar('val/SSIM', ssim_mean, epoch)

            print("             learning rate:º%f" % (optimizer.param_groups[0]['lr']))
            print('             validate loss: {:.7f}'.format(v_loss))
            print('             SAM: {:.6f} | CC: {:.6f} | ERGAS: {:.6f} | SSIM: {:.6f} | PSNR: {:.6f}'.format(
                sam_mean, cc_mean, ergas_mean, ssim_mean, psnr_mean
            ))
    writer.close()  # close tensorboard

# 测试函数：对测试数据进行推理并保存结果
def test():
    """Test the 300th checkpoint on the test split and print five metrics.

    The test set in this workspace stores full scenes in flat HDF5 tensors,
    so inference is done patch-by-patch to avoid GPU OOM.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this test script.")

    test_set = DatasetFromHdf5(TEST_DATASET)
    print(torch.cuda.get_device_name(0))
    print(f"Testing samples: {len(test_set)}")

    # Full-scene test data are 512x512 with LRHSI downsampled by 4.
    gt_patch_size = 64
    scale = scale_factor
    lr_patch_size = gt_patch_size // scale
    num_testing = len(test_set)

    model = build_model(model_name, num_channel=num_hsi_channels,
                        msi_channels=num_msi_channels,
                        scale_factor=scale_factor).cuda().eval()
    path_checkpoint = os.environ.get("TEST_CKPT", model_folder + "300.pth")
    checkpoint = torch.load(path_checkpoint, map_location='cpu')
    state_dict = checkpoint.get('net', checkpoint)
    model.load_state_dict(state_dict, strict=True)
    print(f"Loaded checkpoint: {path_checkpoint}")

    scene_psnr, scene_sam, scene_cc, scene_ergas, scene_ssim = [], [], [], [], []

    for idx in range(num_testing):
        GT, LRHSI, HRMSI = test_set[idx]
        GT = GT.cuda().float()
        LRHSI = LRHSI.cuda().float()
        HRMSI = HRMSI.cuda().float()

        _, h_gt, w_gt = GT.shape
        out_full = torch.zeros_like(GT)
        up_full = torch.zeros_like(GT)
        hp_full = torch.zeros_like(GT)

        with torch.no_grad():
            for top in range(0, h_gt, gt_patch_size):
                for left in range(0, w_gt, gt_patch_size):
                    gt_patch = GT[:, top:top + gt_patch_size, left:left + gt_patch_size].unsqueeze(0)
                    lr_top = top // scale
                    lr_left = left // scale
                    lr_patch = LRHSI[:, lr_top:lr_top + lr_patch_size, lr_left:lr_left + lr_patch_size].unsqueeze(0)
                    hr_patch = HRMSI[:, top:top + gt_patch_size, left:left + gt_patch_size].unsqueeze(0)

                    output_patch, up_patch, hp_patch = model(lr_patch, hr_patch)
                    output_patch = output_patch.clamp(0, 1)
                    out_full[:, top:top + gt_patch_size, left:left + gt_patch_size] = output_patch[0]
                    up_full[:, top:top + gt_patch_size, left:left + gt_patch_size] = up_patch[0]
                    hp_full[:, top:top + gt_patch_size, left:left + gt_patch_size] = hp_patch[0]

        out_b = out_full.unsqueeze(0)
        gt_b = GT.unsqueeze(0)
        psnr_v = float(_psnr_batch(out_b, gt_b).item())
        sam_v = float(_sam_batch(out_b, gt_b).item())
        cc_v = float(_cc_batch(out_b, gt_b).item())
        ergas_v = float(_ergas_batch(out_b, gt_b).item())
        ssim_v = float(_ssim_batch(out_b, gt_b).item())

        scene_psnr.append(psnr_v)
        scene_sam.append(sam_v)
        scene_cc.append(cc_v)
        scene_ergas.append(ergas_v)
        scene_ssim.append(ssim_v)

        print(
            f"Scene {idx + 1:02d}/{num_testing}: "
            f"PSNR={psnr_v:.4f} | SAM={sam_v:.4f} | CC={cc_v:.4f} | ERGAS={ergas_v:.4f} | SSIM={ssim_v:.4f}"
        )

    print("==================== Test Set Average ====================")
    print(f"PSNR : {float(np.mean(scene_psnr)):.4f}")
    print(f"SAM  : {float(np.mean(scene_sam)):.4f}")
    print(f"CC   : {float(np.mean(scene_cc)):.4f}")
    print(f"ERGAS: {float(np.mean(scene_ergas)):.4f}")
    print(f"SSIM : {float(np.mean(scene_ssim)):.4f}")
###################################################################
# ------------------- 主函数 -------------------
###################################################################
if __name__ == "__main__":
    mp.freeze_support()
    train_or_not = 1  # 0表示不训练，1表示训练
    test_or_not = 0   # 0表示不测试，1表示测试
    resume_or_not = 0  # 0表示不恢复，1表示恢复训练
    # 默认从第500 轮权重继续训练
    resume_ckpt = model_folder + "0.pth"

    if train_or_not:  # 如果选择训练
        init_training_state()
        print(torch.cuda.is_available())  # 检查CUDA是否可用
        print(torch.cuda.get_device_name(0))  # 打印GPU名称
        print(torch.cuda.device_count())  # 打印GPU数量
        print(f"Training setup: batch_size={batch_size}, num_workers={num_workers}, amp={use_amp}")
        # 加载训练数据集
        train_set = DatasetFromHdf5(TRAIN_DATASET)
        train_loader_kwargs = dict(
            num_workers=num_workers,  
            batch_size=batch_size,
            shuffle=True,
            pin_memory=(torch.cuda.is_available() and not IS_WINDOWS),
            drop_last=True,
            persistent_workers=False,
        )
        if num_workers > 0:
            train_loader_kwargs["prefetch_factor"] = 2
        training_data_loader = DataLoader(dataset=train_set, **train_loader_kwargs)  # 创建训练数据加载器
        # 加载验证数据集
        validate_set = DatasetFromHdf5(VAL_DATASET)
        val_loader_kwargs = dict(
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=(torch.cuda.is_available() and not IS_WINDOWS),
            drop_last=False,
            persistent_workers=False,
        )
        if num_workers > 0:
            val_loader_kwargs["prefetch_factor"] = 2
        validate_data_loader = DataLoader(dataset=validate_set, **val_loader_kwargs)  # 创建验证数据加载器
        train(
            training_data_loader,
            validate_data_loader,
            RESUME=bool(resume_or_not),
            resume_ckpt=resume_ckpt,
        )  # 调用训练函数

    if test_or_not:  # 如果选择测试
        print("----------------------------testing-------------------------------")
        test()  # 调用测试函数
