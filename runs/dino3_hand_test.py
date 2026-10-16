import os
import torch
import cv2
import numpy as np
import torchvision.transforms.functional as F
import time


# ===================== 模型定义（与训练一致） =====================
class DINOv3HandKeypoint(torch.nn.Module):
    def __init__(self, num_keypoints=21, device='cuda'):
        super().__init__()
        self.num_keypoints = num_keypoints

        current_dir = os.path.dirname(os.path.abspath(__file__))
        local_dinov3_path = os.path.join(current_dir, 'facebookresearch_dinov3_main')
        self.backbone = torch.hub.load(local_dinov3_path, 'dinov3_vits16', source='local', pretrained=False)
        local_weights_path = "./dinov3_wieghts/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
        if os.path.exists(local_weights_path):
            state_dict = torch.load(local_weights_path, map_location=device, weights_only=False)
            self.backbone.load_state_dict(state_dict, strict=True)
        else:
            raise FileNotFoundError(f"权重文件不存在: {local_weights_path}")

        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(384, 256, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, num_keypoints, kernel_size=1)
        )

    def forward(self, x):
        feats = self.backbone.get_intermediate_layers(x, n=1, reshape=True, norm=True)
        x = feats[-1]
        heatmaps = self.decoder(x)
        return heatmaps


# ===================== 预处理与后处理 =====================
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# 手部骨骼连接关系（21点标准顺序）
HAND_SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4),     # 拇指
    (0, 5), (5, 6), (6, 7), (7, 8),     # 食指
    (0, 9), (9, 10), (10, 11), (11, 12),# 中指
    (0, 13), (13, 14), (14, 15), (15, 16),# 无名指
    (0, 17), (17, 18), (18, 19), (19, 20)# 小指
]


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2]
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


def preprocess(frame, img_size=640):
    """输入BGR图片，返回模型输入tensor + 缩放参数"""
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_letter, ratio, dw, dh = letterbox(img_rgb, (img_size, img_size))
    img_tensor = F.to_tensor(img_letter)
    img_tensor = F.normalize(img_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return img_tensor.unsqueeze(0), ratio, dw, dh


def heatmap_to_kpts(heatmaps, ratio, dw, dh, img_size=640, heatmap_size=160):
    """热图峰值 -> 原图关键点坐标"""
    num_kpts = heatmaps.shape[0]
    kpts = []
    scale = img_size / heatmap_size
    for i in range(num_kpts):
        y_idx, x_idx = np.unravel_index(np.argmax(heatmaps[i]), heatmaps[i].shape)
        x_letter = x_idx * scale
        y_letter = y_idx * scale
        x_orig = (x_letter - dw) / ratio
        y_orig = (y_letter - dh) / ratio
        kpts.append((int(x_orig), int(y_orig)))
    return kpts


def draw_kpts(frame, kpts):
    """绘制关键点和骨骼连线"""
    img = frame.copy()
    # 画连线
    for (i, j) in HAND_SKELETON:
        cv2.line(img, kpts[i], kpts[j], (255, 128, 0), 2)
    # 画关键点
    for x, y in kpts:
        cv2.circle(img, (x, y), 4, (0, 255, 0), -1)
    return img


# ===================== 主程序 =====================
def main():
    # ========== 配置 ==========
    model_weight = "runs/dino3_hand/best.pt"
    img_size = 640
    heatmap_size = 160
    num_keypoints = 21
    device = "cuda" if torch.cuda.is_available() else "cpu"
    source = 0  # 0=摄像头，也可填图片路径如 "test.jpg"

    # 加载模型
    print("正在加载模型...")
    model = DINOv3HandKeypoint(num_keypoints=num_keypoints, device=device)
    model.load_state_dict(torch.load(model_weight, map_location=device))
    model = model.to(device).eval()
    print("模型加载完成！")

    # 判断输入源
    if isinstance(source, str) and os.path.isfile(source):
        # 单张图片测试
        frame = cv2.imread(source)
        if frame is None:
            print(f"无法读取图片: {source}")
            return

        img_tensor, ratio, dw, dh = preprocess(frame, img_size)
        with torch.no_grad():
            pred = model(img_tensor.to(device))
            heatmaps = pred.squeeze(0).cpu().numpy()
        kpts = heatmap_to_kpts(heatmaps, ratio, dw, dh, img_size, heatmap_size)
        result = draw_kpts(frame, kpts)

        cv2.imshow("DINOv3 手部关键点", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        # 摄像头实时测试
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print("摄像头打开失败")
            return

        print("摄像头已启动，按 ESC 退出")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.time()
            img_tensor, ratio, dw, dh = preprocess(frame, img_size)
            with torch.no_grad():
                pred = model(img_tensor.to(device))
                heatmaps = pred.squeeze(0).cpu().numpy()
            kpts = heatmap_to_kpts(heatmaps, ratio, dw, dh, img_size, heatmap_size)
            result = draw_kpts(frame, kpts)
            fps = 1 / (time.time() - t0)

            cv2.putText(result, f"FPS: {fps:.1f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("DINOv3 手部关键点", result)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC退出
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
