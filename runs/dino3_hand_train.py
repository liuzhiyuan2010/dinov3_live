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

        # 缩放+填充
        img, ratio, dw, dh = letterbox(img, (self.img_size, self.img_size))

        # 解析YOLO格式关键点标注
        keypoints = []
        label_path = self.label_paths[idx]
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                line = f.readline().strip().split()
                # 格式: class x_center y_center w h kx1 ky1 kv1 ... kx21 ky21 kv21
                kpts_data = list(map(float, line[5:]))
                for i in range(self.num_keypoints):
                    kx_norm = kpts_data[i * 3]
                    ky_norm = kpts_data[i * 3 + 1]
                    visible = kpts_data[i * 3 + 2]
                    # 归一化 -> 原图绝对坐标
                    kx = kx_norm * w_orig
                    ky = ky_norm * h_orig
                    # 原图 -> letterbox坐标
                    kx = kx * ratio + dw
                    ky = ky * ratio + dh
                    # letterbox -> 热图坐标
                    kx_heat = kx * (self.heatmap_size / self.img_size)
                    ky_heat = ky * (self.heatmap_size / self.img_size)
                    keypoints.append((kx_heat, ky_heat, visible))

        # 生成多通道热图
        heatmaps = np.zeros((self.num_keypoints, self.heatmap_size, self.heatmap_size), dtype=np.float32)
        for i in range(self.num_keypoints):
            kx, ky, visible = keypoints[i]
            if visible > 0:
                heatmaps[i] = generate_heatmap((self.heatmap_size, self.heatmap_size), (kx, ky), self.sigma)

        # 图片预处理：和DINOv3输入标准对齐
        img_tensor = F.to_tensor(img)
        img_tensor = F.normalize(img_tensor, mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
        heatmap_tensor = torch.from_numpy(heatmaps)

        return img_tensor, heatmap_tensor




# ===================== 模型定义 =====================
class DINOv3HandKeypoint(nn.Module):
    def __init__(self, num_keypoints=21, freeze_backbone=True, device='cuda'):
        super().__init__()
        self.num_keypoints = num_keypoints

        # 加载本地DINOv3骨干，和你原有SDK路径完全对齐
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
        # 输入: [B, 384, 40, 40] (640输入下)
        # 输出: [B, 21, 160, 160]
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(384, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_keypoints, kernel_size=1)
        )

    def forward(self, x):
        # 提取DINOv3空间特征
        feats = self.backbone.get_intermediate_layers(x, n=1, reshape=True, norm=True)
        x = feats[-1]  # [B, 384, H/16, W/16]
        heatmaps = self.decoder(x)
        return heatmaps


# ===================== 训练&验证 =====================
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc="Train")
    for imgs, heatmaps in pbar:
        imgs = imgs.to(device)
        heatmaps = heatmaps.to(device)

        optimizer.zero_grad()
        pred = model(imgs)
        loss = criterion(pred, heatmaps)
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

    for imgs, heatmaps_gt in tqdm(dataloader, desc="Val"):
        imgs = imgs.to(device)
        heatmaps_gt = heatmaps_gt.to(device)
        pred = model(imgs)
        loss = criterion(pred, heatmaps_gt)
        total_loss += loss.item()

        # 计算PCK指标
        B, K, H, W = pred.shape
        diag = np.sqrt(H ** 2 + W ** 2)
        for b in range(B):
            for k in range(K):
                gt_hm = heatmaps_gt[b, k]
                if gt_hm.max() == 0:
                    continue
                # 真实坐标
                y_gt, x_gt = torch.unravel_index(torch.argmax(gt_hm), gt_hm.shape)
                # 预测坐标
                pred_hm = pred[b, k]
                y_pred, x_pred = torch.unravel_index(torch.argmax(pred_hm), pred_hm.shape)
                # 距离计算
                dist = torch.sqrt((x_pred - x_gt) ** 2 + (y_pred - y_gt) ** 2).item()
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
        "data_yaml": "data.yaml",       # 你原有的数据集配置文件
        "img_size": 640,                # 输入图片尺寸，和原YOLO训练一致
        "heatmap_size": 160,            # 输出热图尺寸 (1/4原图)
        "sigma": 3,                     # 高斯热图标准差
        "num_keypoints": 21,            # 手部21关键点
        "freeze_backbone": True,        # True=冻结骨干训头，False=全量微调
        "epochs": 10,
        "batch_size": 16,
        "lr": 0.001,                    # 冻结骨干时用1e-3，微调建议1e-5
        "weight_decay": 0.0005,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "save_dir": "runs/dino3_hand",
        "patience": 20,
        "pck_threshold": 0.2
    }

    # 读取数据集配置
    with open(cfg["data_yaml"], "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    data_root = data_cfg.get("path", "./")

    # 构建数据集
    train_set = HandKeypointDataset(data_root, "train", cfg["img_size"], cfg["heatmap_size"], cfg["sigma"], cfg["num_keypoints"], is_train=True)
    val_set = HandKeypointDataset(data_root, "val", cfg["img_size"], cfg["heatmap_size"], cfg["sigma"], cfg["num_keypoints"], is_train=False)
    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=cfg["batch_size"], shuffle=False, num_workers=8, pin_memory=True)
    print(f"训练集: {len(train_set)} 张 | 验证集: {len(val_set)} 张")

    # 初始化模型
    model = DINOv3HandKeypoint(
        num_keypoints=cfg["num_keypoints"],
        freeze_backbone=cfg["freeze_backbone"],
        device=cfg["device"]
    ).to(cfg["device"])

    # 损失与优化器
    criterion = nn.MSELoss()
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(trainable_params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"], eta_min=cfg["lr"] * 0.01)

    # 创建保存目录
    os.makedirs(cfg["save_dir"], exist_ok=True)
    best_pck = 0.0
    no_improve = 0

    print("=" * 50)
    print("开始训练 DINOv3 手部关键点模型")
    print("=" * 50)

    for epoch in range(cfg["epochs"]):
        print(f"\nEpoch {epoch + 1}/{cfg['epochs']}")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, cfg["device"])
        val_loss, val_pck = validate(model, val_loader, criterion, cfg["device"], cfg["pck_threshold"])
        scheduler.step()

        print(f"训练损失: {train_loss:.4f} | 验证损失: {val_loss:.4f} | 验证PCK@{cfg['pck_threshold']}: {val_pck:.4f}")

        # 保存最佳模型
        if val_pck > best_pck:
            best_pck = val_pck
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(cfg["save_dir"], "best.pt"))
            print(f"✅ 最佳模型已更新，PCK: {best_pck:.4f}")
        else:
            no_improve += 1

        # 保存最新模型
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
