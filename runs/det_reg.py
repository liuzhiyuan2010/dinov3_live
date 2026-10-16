from ultralytics import YOLO
import cv2
import traceback
import os
import torch
import numpy as np
import torchvision.transforms.functional as F
import time
import torch.nn as nn

# ===================== DINOv3 关键点模块（原代码原样保留） =====================
class DINOv3HandKeypoint(torch.nn.Module):
    def __init__(self, num_keypoints=21, device='cuda', load_dino_pretrain=True):
        super().__init__()
        self.num_keypoints = num_keypoints

        current_dir = os.path.dirname(os.path.abspath(__file__))
        local_dinov3_path = os.path.join(current_dir, 'facebookresearch_dinov3_main')
        self.backbone = torch.hub.load(local_dinov3_path, 'dinov3_vits16', source='local', pretrained=False)
        
        # 只有训练阶段才加载DINO原生预训练权重
        if load_dino_pretrain:
            local_weights_path = "./dinov3_wieghts/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
            if os.path.exists(local_weights_path):
                state_dict = torch.load(local_weights_path, map_location=device, weights_only=False)
                self.backbone.load_state_dict(state_dict, strict=True)
            else:
                raise FileNotFoundError(f"权重文件不存在: {local_weights_path}")

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


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

HAND_SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20)
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
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_letter, ratio, dw, dh = letterbox(img_rgb, (img_size, img_size))
    img_tensor = F.to_tensor(img_letter)
    img_tensor = F.normalize(img_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return img_tensor.unsqueeze(0), ratio, dw, dh


def get_subpixel_peak(heatmap):
    # heatmap: [H,W]
     # 如果是numpy数组，先转成tensor
    if isinstance(heatmap, np.ndarray):
        heatmap = torch.from_numpy(heatmap)
        
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


def heatmap_to_kpts(heatmaps, ratio, dw, dh, img_size=640, heatmap_size=160):
    num_kpts = heatmaps.shape[0]
    kpts = []
    scale = img_size / heatmap_size
    for i in range(num_kpts):
        #y_idx, x_idx = np.unravel_index(np.argmax(heatmaps[i]), heatmaps[i].shape)
        y_idx, x_idx = get_subpixel_peak(heatmaps[i])
        x_letter = x_idx * scale
        y_letter = y_idx * scale
        x_orig = (x_letter - dw) / ratio
        y_orig = (y_letter - dh) / ratio
        kpts.append((int(x_orig), int(y_orig)))
    return kpts

def heatmap_to_kpts0(heatmaps, ratio, dw, dh, img_size=640, heatmap_size=160):
    num_kpts = heatmaps.shape[0]
    kpts = []
    scale = img_size / heatmap_size
    for i in range(num_kpts):
        hm = heatmaps[i]
        # 高斯平滑去噪，消除杂散小峰值
        hm = cv2.GaussianBlur(hm, (3, 3), 0.8)
        
        y_idx, x_idx = np.unravel_index(np.argmax(hm), hm.shape)
        # 亚像素偏移修正，提升定位精度
        if 1 < x_idx < hm.shape[1] - 1 and 1 < y_idx < hm.shape[0] - 1:
            dx = 0.5 * (hm[y_idx, x_idx + 1] - hm[y_idx, x_idx - 1])
            dy = 0.5 * (hm[y_idx + 1, x_idx] - hm[y_idx - 1, x_idx])
            x_idx += dx
            y_idx += dy
        
        x_letter = x_idx * scale
        y_letter = y_idx * scale
        x_orig = (x_letter - dw) / ratio
        y_orig = (y_letter - dh) / ratio
        kpts.append((int(round(x_orig)), int(round(y_orig))))
    return kpts

def draw_kpts_on_subimg(sub_img, kpts_sub):
    """在裁剪的手部小图画关键点（调试用）"""
    img = sub_img.copy()
    for (i, j) in HAND_SKELETON:
        cv2.line(img, kpts_sub[i], kpts_sub[j], (255, 128, 0), 2)
    for x, y in kpts_sub:
        cv2.circle(img, (x, y), 4, (0, 255, 0), -1)
    return img


# ===================== 全局初始化两个模型（只加载一次，提速） =====================
print("=== 程序开始启动 ===")
# 1. YOLO手部检测模型
yolo_model_path = r"hand_det_best.pt"
print(f"加载YOLO检测模型: {yolo_model_path}")
yolo_model = YOLO(yolo_model_path)

# 2. DINOv3关键点模型
KP_WEIGHT = "runs/dino3_hand/best.pt"
KP_IMG_SIZE = 192
KP_HEATMAP_SIZE = 48
KP_NUM = 21
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"加载DINOv3关键点模型: {KP_WEIGHT}")
kp_model = DINOv3HandKeypoint(num_keypoints=KP_NUM, device=device,load_dino_pretrain=False)
kp_model.load_state_dict(torch.load(KP_WEIGHT, map_location=device, weights_only=False))
kp_model = kp_model.to(device).eval()
# 推理禁用梯度
torch.set_grad_enabled(False)

# 外扩像素：防止YOLO框切到手部边缘关键点
PADDING_RATIO = 0.2

# ===================== 主摄像头循环 =====================
try:
    print("正在打开摄像头 id=0")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("❌ 摄像头打开失败！尝试更换设备ID 1/2")
        exit(1)
    print("✅ 摄像头成功打开")

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        frame_count += 1
        if not ret:
            print(f"❌ 第{frame_count}帧读取失败，退出")
            break

        # YOLO推理检测手
        results = yolo_model.predict(
            source=frame,
            save=False,
            imgsz=640,
            conf=0.45,
            device=0,
            half=True
        )

        draw_frame = frame.copy()
        img_h, img_w = frame.shape[:2]

        for res in results:
            boxes = res.boxes
            if boxes is None or len(boxes) == 0:
                continue

            # 遍历每一个手检测框
            xyxy_batch = boxes.xyxy.cpu().numpy()
            conf_batch = boxes.conf.cpu().numpy()
            cls_batch = boxes.cls.cpu().numpy()

            for idx, xyxy in enumerate(xyxy_batch):
                # 原始xyxy检测框
                x1_raw, y1_raw, x2_raw, y2_raw = xyxy
                # 框宽、框高
                bbox_w = x2_raw - x1_raw
                bbox_h = y2_raw - y1_raw

                # 和训练脚本完全一致：宽按宽比例扩，高按高比例扩
                pad_w = bbox_w * PADDING_RATIO
                pad_h = bbox_h * PADDING_RATIO

                # 计算外扩后坐标 + 边界钳位（和训练代码公式一模一样）
                x1 = int(max(0, x1_raw - pad_w))
                y1 = int(max(0, y1_raw - pad_h))
                x2 = int(min(img_w, x2_raw + pad_w))
                y2 = int(min(img_h, y2_raw + pad_h))





                # 裁剪出手部区域
                hand_crop = frame[y1:y2, x1:x2]
                if hand_crop.size == 0:
                    continue

                # 送入关键点模型推理
                tensor, ratio, dw, dh = preprocess(hand_crop, img_size=KP_IMG_SIZE)
                pred = kp_model(tensor.to(device))
                heatmaps = pred.squeeze(0).cpu().numpy()
                # 得到相对于裁剪图的关键点
                kpts_crop = heatmap_to_kpts(heatmaps, ratio, dw, dh, KP_IMG_SIZE, KP_HEATMAP_SIZE)

                # 把裁剪图内关键点映射回原图坐标
                kpts_origin = []
                for (cx, cy) in kpts_crop:
                    ox = cx + x1
                    oy = cy + y1
                    kpts_origin.append((ox, oy))

                # 在原图绘制手部框 + 骨骼关键点
                # 画YOLO检测框
                cv2.rectangle(draw_frame, (x1, y1), (x2, y2), (0,0,255), 2)
                cv2.putText(draw_frame, f"hand {conf_batch[idx]:.2f}", (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

                # 绘制21点骨架
                for (p1, p2) in HAND_SKELETON:
                    cv2.line(draw_frame, kpts_origin[p1], kpts_origin[p2], (0,200,255), 2)
                for (px, py) in kpts_origin:
                    cv2.circle(draw_frame, (px, py), 3, (0,255,0), -1)

        cv2.imshow("YOLO Detect + DINOv3 Hand Keypoint", draw_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("⌨️ 按下q退出")
            break

except Exception as main_err:
    print("\n❌ 程序异常：")
    traceback.print_exc()
finally:
    print("\n释放资源...")
    if 'cap' in locals():
        cap.release()
    cv2.destroyAllWindows()
    print("程序结束")
