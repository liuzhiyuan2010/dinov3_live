import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import torchvision.transforms.functional as F
from tqdm import tqdm

def get_subpixel_peak(heatmap):
    # heatmap: [H,W]
    y, x = torch.unravel_index(torch.argmax(heatmap), heatmap.shape)
    h, w = heatmap.shape
    # 边界保护
    y = torch.clamp(y, 1, h-2)
    x = torch.clamp(x, 1, w-2)
    # 取3邻域做二次插值
    dx = (heatmap[y, x+1] - heatmap[y, x-1]) / 2.0
    dy = (heatmap[y+1, x] - heatmap[y-1, x]) / 2.0
    x_float = x + dx
    y_float = y + dy
    return y_float.item(), x_float.item()


# ===================== 工具函数 =====================
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """保持长宽比缩放+padding，和YOLO预处理对齐"""
    shape = img.shape[:2]  # [height, width]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, dw, dh


def generate_heatmap(heatmap_size, center, sigma=3):
    """生成单关键点高斯热图"""
    h, w = heatmap_size
    x, y = center
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    heatmap = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
    heatmap[heatmap < np.finfo(np.float32).eps] = 0
    return heatmap


# ===================== 损失函数 =====================
class FocalMSELoss(nn.Module):
    """
    热图专用Focal-MSE损失（逐像素输出，外部加权求平均）
    gamma=0 等价于原生逐像素MSE
    """
    def __init__(self, gamma=2.0, alpha=1.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, target):
        diff = torch.abs(pred - target)
        # 调制因子：误差越小的像素权重越低
        weight = torch.pow(diff, self.gamma)
        loss = self.alpha * weight * torch.square(diff)
        return loss  # 返回逐像素损失，不做全局平均


# ===================== 数据集类 =====================
class HandKeypointDataset(Dataset):
    def __init__(self, data_root, split, img_size=640, heatmap_size=160, sigma=3, num_keypoints=21, is_train=False):
        self.img_size = img_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.num_keypoints = num_keypoints
        self.is_train = is_train
        self.IMAGENET_MEAN = (0.485, 0.456, 0.406)
        self.IMAGENET_STD = (0.229, 0.224, 0.225)

        # 读取图片和标注路径
        img_dir = os.path.join(data_root, f'images/{split}')
        label_dir = os.path.join(data_root, f'labels/{split}')
        self.img_paths = []
        self.label_paths = []
        for f in os.listdir(img_dir):
            if f.endswith(('.jpg', '.png', '.jpeg')):
                self.img_paths.append(os.path.join(img_dir, f))
                label_name = os.path.splitext(f)[0] + '.txt'
                self.label_paths.append(os.path.join(label_dir, label_name))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # 读取图片
        img = cv2.imread(self.img_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = img.shape[:2]

        # 先解析原图关键点绝对坐标 + 可见性
        keypoints_orig = []
        vis_list = []
        label_path = self.label_paths[idx]
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                line = f.readline().strip().split()
                kpts_data = list(map(float, line[5:]))
                for i in range(self.num_keypoints):
                    kx_norm = kpts_data[i * 3]
                    ky_norm = kpts_data[i * 3 + 1]
                    visible = kpts_data[i * 3 + 2]
                    # 归一化 -> 原图绝对坐标
                    kx = kx_norm * w_orig
                    ky = ky_norm * h_orig
                    keypoints_orig.append([kx, ky])
                    vis_list.append(visible)
        keypoints_orig = np.array(keypoints_orig, dtype=np.float32)

        # ========== 仅训练阶段启用数据增强 ==========
        if self.is_train:
            # 色彩增强保留不变
            if np.random.random() < 0.5:
                brightness = np.random.uniform(-0.12, 0.12) * 255
                img = np.clip(img.astype(np.float32) + brightness, 0, 255).astype(np.uint8)
            if np.random.random() < 0.5:
                contrast = np.random.uniform(0.85, 1.15)
                mean = img.mean()
                img = np.clip((img.astype(np.float32) - mean) * contrast + mean, 0, 255).astype(np.uint8)
            if np.random.random() < 0.5:
                sat_scale = np.random.uniform(0.75, 1.25)
                hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[..., 1] = np.clip(hsv[..., 1] * sat_scale, 0, 255)
                img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            if np.random.random() < 0.4:
                gamma = np.random.uniform(0.8, 1.2)
                inv_gamma = 1.0 / gamma
                table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype(np.uint8)
                img = cv2.LUT(img, table)

            # 水平翻转：仅反转X，删除关键点互换
            if np.random.random() < 0.5:
                img = cv2.flip(img, 1)
                keypoints_orig[:, 0] = w_orig - keypoints_orig[:, 0]

            if np.random.random() < 0.05:
                ksize = np.random.choice([3, 5])
                img = cv2.GaussianBlur(img, (ksize, ksize), 0)
            if np.random.random() < 0.02:
                noise = np.random.normal(0, 8, img.shape).astype(np.int16)
                img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


        # 缩放+填充（letterbox，原逻辑完全不变）
        img, ratio, dw, dh = letterbox(img, (self.img_size, self.img_size))

        # 坐标转换：原图 → letterbox → 热图
        keypoints_heat = []
        for i in range(self.num_keypoints):
            kx, ky = keypoints_orig[i]
            # 原图 -> letterbox坐标
            kx = kx * ratio + dw
            ky = ky * ratio + dh
            # letterbox -> 热图坐标
            kx_heat = kx * (self.heatmap_size / self.img_size)
            ky_heat = ky * (self.heatmap_size / self.img_size)
            # 新增两行钳位
            kx_heat = np.clip(kx_heat, 0, self.heatmap_size - 1)
            ky_heat = np.clip(ky_heat, 0, self.heatmap_size - 1)

            # 超出热图范围标记为不可见
            if kx_heat < 0 or kx_heat >= self.heatmap_size or ky_heat < 0 or ky_heat >= self.heatmap_size:
                vis_list[i] = 0
            keypoints_heat.append((kx_heat, ky_heat, vis_list[i]))

        # 生成多通道热图
        heatmaps = np.zeros((self.num_keypoints, self.heatmap_size, self.heatmap_size), dtype=np.float32)
        for i in range(self.num_keypoints):
            kx, ky, visible = keypoints_heat[i]
            heatmaps[i] = generate_heatmap((self.heatmap_size, self.heatmap_size), (kx, ky), self.sigma)

        # 图片预处理
        img_tensor = F.to_tensor(img)
        img_tensor = F.normalize(img_tensor, mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
        heatmap_tensor = torch.from_numpy(heatmaps)
        vis_tensor = torch.tensor(vis_list, dtype=torch.float32)

        return img_tensor, heatmap_tensor, vis_tensor


# ===================== 模型定义 =====================
class DINOv3HandKeypoint(nn.Module):
    def __init__(self, num_keypoints=21, freeze_backbone=True, device='cuda'):
        super().__init__()
        self.num_keypoints = num_keypoints

        # 加载本地DINOv3骨干
        current_dir = os.path.dirname(os.path.abspath(__file__))
        local_dinov3_path = os.path.join(current_dir, 'facebookresearch_dinov3_main')
        self.backbone = torch.hub.load(local_dinov3_path, 'dinov3_vits16', source='local', pretrained=False)
        local_weights_path = "./dinov3_wieghts/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
        if os.path.exists(local_weights_path):
            state_dict = torch.load(local_weights_path, map_location=device, weights_only=False)
            self.backbone.load_state_dict(state_dict, strict=True)
            print("✅ DINOv3预训练权重加载成功")
        else:
            raise FileNotFoundError(f"权重文件不存在: {local_weights_path}")

        # 冻结骨干
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("🔒 DINOv3骨干已冻结，仅训练上采样关键点头")

        # 上采样解码头：1/16 -> 1/4 分辨率输出热图
        self.decoder = nn.Sequential(
            # 1/16 -> 1/8
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(384, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 1/8 -> 1/4
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_keypoints, 1),
            nn.Sigmoid()
        )

        self.w1 = nn.Parameter(torch.tensor(1.0))
        self.w2 = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        feats = self.backbone.get_intermediate_layers(x, n=[3, 7, 11], reshape=True, norm=True)
        f4, f8, f12 = feats
        x = self.w1 * f4 + self.w2 * f8 + f12
        heatmaps = self.decoder(x)
        return heatmaps


# ===================== 训练&验证 =====================
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc="Train")
    for imgs, heatmaps, vis_mask in pbar:
        imgs = imgs.to(device)
        heatmaps = heatmaps.to(device)
        vis_mask = vis_mask.to(device)

        optimizer.zero_grad()
        pred = model(imgs)

        # 可见性加权损失：可见点全额监督，遮挡点弱监督（0.15权重）
        B, K, H, W = pred.shape
        vis_weight = vis_mask.view(B, K, 1, 1).expand(B, K, H, W)
        vis_weight = torch.where(vis_weight > 0, 1.0, 0.05)  # 遮挡点保留15%损失权重

        # 逐像素损失 + 加权 + 全局平均
        pixel_loss = criterion(pred, heatmaps)
        loss = torch.mean(pixel_loss * vis_weight)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / len(dataloader)


@torch.no_grad()
def validate(model, dataloader, criterion, device, pck_threshold=0.2):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_visible = 0

    for imgs, heatmaps_gt, vis_mask in tqdm(dataloader, desc="Val"):
        imgs = imgs.to(device)
        heatmaps_gt = heatmaps_gt.to(device)
        vis_mask = vis_mask.to(device)

        pred = model(imgs)

        # 验证损失与训练保持完全一致的加权逻辑
        B, K, H, W = pred.shape
        vis_weight = vis_mask.view(B, K, 1, 1).expand(B, K, H, W)
        vis_weight = torch.where(vis_weight > 0, 1.0, 0.05)
        
        pixel_loss = criterion(pred, heatmaps_gt)
        loss = torch.mean(pixel_loss * vis_weight)
        total_loss += loss.item()

        # 计算PCK指标（仅统计可见关键点）
        diag = np.sqrt(H ** 2 + W ** 2)
        for b in range(B):
            for k in range(K):
                if vis_mask[b, k] <= 0:
                    continue
                # 真实坐标
                gt_hm = heatmaps_gt[b, k]
                #y_gt, x_gt = torch.unravel_index(torch.argmax(gt_hm), gt_hm.shape)
                y_gt, x_gt = get_subpixel_peak(gt_hm)

                # 预测坐标
                pred_hm = pred[b, k]
                #y_pred, x_pred = torch.unravel_index(torch.argmax(pred_hm), pred_hm.shape)
                y_pred, x_pred = get_subpixel_peak(pred_hm)
                
                # 距离计算
                #dist = torch.sqrt((x_pred - x_gt) ** 2 + (y_pred - y_gt) ** 2).item()
                dist = np.sqrt((x_pred - x_gt) ** 2 + (y_pred - y_gt) ** 2)

                if dist < diag * pck_threshold:
                    total_correct += 1
                total_visible += 1

    avg_loss = total_loss / len(dataloader)
    pck = total_correct / total_visible if total_visible > 0 else 0.0
    return avg_loss, pck


# ===================== 主函数 =====================
def main():
    # ========== 配置参数 ==========
    cfg = {
        "data_yaml": "data3.yaml",
        "img_size": 192,
        "heatmap_size": 48,
        "sigma": 3.5,              # 由2调至2.5，增加标签容错，减少串指
        "num_keypoints": 21,
        "freeze_backbone": True,
        "epochs": 30,
        "batch_size": 32,
        "lr": 0.0003,               # 冻结骨干用1e-3，全量微调建议1e-5
        "weight_decay": 0.0005,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "save_dir": "runs/dino3_hand",
        "patience": 20,
        "pck_threshold": 0.2,
        "num_workers": 0,          # Windows设0，Linux可设4~8
        "resume": False,           # True=自动断点续训，False=从头训练
        "loss_type": "mse",        # 先用原生MSE稳定训练，后续可切focal_mse
        "focal_gamma": 1.5         # focal模式下gamma不超过1.5，避免梯度爆炸
    }

    # 读取数据集配置
    with open(cfg["data_yaml"], "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    data_root = data_cfg.get("path", "./")

    # 构建数据集
    train_set = HandKeypointDataset(data_root, "train", cfg["img_size"], cfg["heatmap_size"], cfg["sigma"], cfg["num_keypoints"], is_train=True)
    val_set = HandKeypointDataset(data_root, "val", cfg["img_size"], cfg["heatmap_size"], cfg["sigma"], cfg["num_keypoints"], is_train=False)
    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"], pin_memory=True)
    print(f"训练集: {len(train_set)} 张 | 验证集: {len(val_set)} 张")

    # 初始化模型
    model = DINOv3HandKeypoint(
        num_keypoints=cfg["num_keypoints"],
        freeze_backbone=cfg["freeze_backbone"],
        device=cfg["device"]
    ).to(cfg["device"])

    # 损失函数（统一使用逐像素输出，外部加权求平均）
    if cfg["loss_type"] == "focal_mse":
        criterion = FocalMSELoss(gamma=cfg["focal_gamma"])
        print(f"📉 使用 FocalMSE 损失，gamma={cfg['focal_gamma']}")
    else:
        criterion = nn.MSELoss(reduction='none')  # 逐像素输出，配合可见性加权
        print("📉 使用原生 MSE 损失（遮挡点弱监督加权）")

    # 优化器与学习率调度
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(trainable_params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"], eta_min=cfg["lr"] * 0.01)

    # 创建保存目录
    os.makedirs(cfg["save_dir"], exist_ok=True)

    # ========== 断点续训加载 ==========
    start_epoch = 0
    best_pck = 0.0
    no_improve = 0
    last_ckpt_path = os.path.join(cfg["save_dir"], "last_ckpt.pth")

    if cfg["resume"] and os.path.exists(last_ckpt_path):
        ckpt = torch.load(last_ckpt_path, map_location=cfg["device"], weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"]
        best_pck = ckpt["best_pck"]
        no_improve = ckpt["no_improve"]
        print(f"🔄 断点续训成功 | 起始Epoch: {start_epoch} | 历史最佳PCK: {best_pck:.4f}")
    else:
        # 兼容旧版纯权重best.pt的续训（无优化器状态）
        old_best_path = os.path.join(cfg["save_dir"], "best.pt")
        if cfg["resume"] and os.path.exists(old_best_path):
            model.load_state_dict(torch.load(old_best_path, map_location=cfg["device"], weights_only=False))
            print(f"🔄 加载旧版best.pt权重继续训练（优化器重新初始化）")
        else:
            print("🆕 无历史断点，从头开始训练")

    print("=" * 50)
    print("开始训练 DINOv3 手部关键点模型")
    print("=" * 50)

    for epoch in range(start_epoch, cfg["epochs"]):
        print(f"\nEpoch {epoch + 1}/{cfg['epochs']}")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, cfg["device"])
        val_loss, val_pck = validate(model, val_loader, criterion, cfg["device"], cfg["pck_threshold"])
        scheduler.step()

        print(f"训练损失: {train_loss:.4f} | 验证损失: {val_loss:.4f} | 验证PCK@{cfg['pck_threshold']}: {val_pck:.4f}")

        # 保存最佳模型快照
        if val_pck > best_pck:
            best_pck = val_pck
            no_improve = 0
            best_ckpt = {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_pck": best_pck,
                "no_improve": no_improve,
                "cfg": cfg
            }
            torch.save(best_ckpt, os.path.join(cfg["save_dir"], "best_ckpt.pth"))
            # 同时存一份纯权重，方便推理直接加载
            torch.save(model.state_dict(), os.path.join(cfg["save_dir"], "best.pt"))
            print(f"✅ 最佳模型已更新，PCK: {best_pck:.4f}")
        else:
            no_improve += 1

        # 保存最新断点快照
        last_ckpt = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_pck": best_pck,
            "no_improve": no_improve,
            "cfg": cfg
        }
        torch.save(last_ckpt, os.path.join(cfg["save_dir"], "last_ckpt.pth"))
        torch.save(model.state_dict(), os.path.join(cfg["save_dir"], "last.pt"))

        # 早停
        if no_improve >= cfg["patience"]:
            print(f"\n早停触发：连续 {cfg['patience']} 轮无精度提升")
            break

    print("\n" + "=" * 50)
    print(f"训练完成！最佳验证 PCK: {best_pck:.4f}")
    print(f"模型保存路径: {cfg['save_dir']}/best.pt")
    print("=" * 50)


if __name__ == "__main__":
    main()
