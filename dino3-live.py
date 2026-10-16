import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import numpy as np
from PIL import Image
import os
import cv2
import time

class DINOv3Pipeline:
    def __init__(self, image_size=640, patch_size=16, device='cuda', KEEP_N=20):
        self.image_size = image_size
        self.patch_size = patch_size
        self.KEEP_N = KEEP_N
        self.IMAGENET_MEAN = (0.485, 0.456, 0.406)
        self.IMAGENET_STD = (0.229, 0.224, 0.225)
        self.device = device
        self.model = self.load_model()

        # 去偏基底缓存（解决每帧随机抖动核心）
        self._base_B = None       # 预计算好的正交基底 [Dim, k]
        self._cached_h_patch = None
        self._cached_w_patch = None

    def image_transform(self, image):
        w, h = image.size
        h_patches = int(self.image_size / self.patch_size)
        w_patches = int((w * self.image_size) / (h * self.patch_size))
        target_h = h_patches * self.patch_size
        target_w = w_patches * self.patch_size
        image_resized = F.to_tensor(F.resize(image, (target_h, target_w)))
        image_resized_norm = F.normalize(image_resized, mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
        return image_resized_norm, h_patches, w_patches, target_w, target_h

    def lock_backbone(self):
        for p in self.model.parameters():
            p.requires_grad = False

    def load_model(self):
        print(f"Using device: {self.device}")
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            local_dinov3_path = os.path.join(current_dir, 'facebookresearch_dinov3_main')
            model = torch.hub.load(local_dinov3_path, 'dinov3_vits16', source='local', pretrained=False)
            local_weights_path = "./dinov3_wieghts/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
            if os.path.exists(local_weights_path):
                state_dict = torch.load(local_weights_path, map_location=self.device, weights_only=False)
                model.load_state_dict(state_dict, strict=True)
                model.to(self.device)
                print("✅ 本地权重加载成功！")
                return model
            else:
                print(f"❌ 找不到本地权重文件: {local_weights_path}")
                return None
        except Exception as e:
            print(f"模型加载错误: {e}")
            return None

    def get_mask_grid(self, mask_img, h_patches, w_patches, patch_size=16, MASK_THRESHOLD=128):
        target_h = h_patches * patch_size
        target_w = w_patches * patch_size
        mask_resized = mask_img.resize((target_w, target_h), resample=Image.NEAREST)
        mask_tensor = F.to_tensor(mask_resized)
        mask_binary = (mask_tensor > (MASK_THRESHOLD / 255.0)).float()
        mask_grid = mask_img.resize((w_patches, h_patches), resample=Image.NEAREST)
        mask_grid_tensor = F.to_tensor(mask_grid)
        mask_binary = (mask_grid_tensor > (MASK_THRESHOLD / 255.0)).float()
        return mask_binary

    def extract_features(self, image):
        self.model.eval()
        self.lock_backbone()
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        preproc_img, h_patches, w_patches, target_w, target_h = self.image_transform(image)
        input_tensor = preproc_img.unsqueeze(0).to(self.device)
        with torch.no_grad():
            feats = self.model.get_intermediate_layers(input_tensor, n=1, reshape=True, norm=True)
        dense_features = feats[-1].squeeze(0)
        return dense_features, h_patches, w_patches, target_w, target_h

    def _build_fixed_basis(self, h_patch, w_patch, orig_w, orig_h):
        """
        仅在分辨率变化时执行一次：生成固定噪声基底 + SVD计算B，缓存复用
        """
        print(f"🔨 生成固定去偏基底，网格尺寸 H={h_patch}, W={w_patch}")
        # 固定随机种子，噪声图完全固定，消除帧间随机性
        rng = np.random.Generator(np.random.PCG64(seed=42))
        rand_img = rng.integers(0, 255, (orig_h, orig_w, 3), dtype=np.uint8)
        base_ff, _, _, _, _ = self.extract_features(rand_img)
        C, H, W = base_ff.shape
        features_np = base_ff.permute(1, 2, 0).reshape(-1, C).cpu().numpy()
        U, S, Vt = np.linalg.svd(features_np, full_matrices=False)
        k = self.KEEP_N
        B_np = Vt[:k, :].T
        self._base_B = torch.from_numpy(B_np).to(self.device).float()
        # 缓存当前分辨率
        self._cached_h_patch = h_patch
        self._cached_w_patch = w_patch

    def debias(self, orig_features, h_patch, w_patch, orig_w, orig_h):
        """
        集成到类内的去偏方法，自动复用固定基底，无帧间抖动
        :param orig_features: [C, H, W] 原始DINO特征
        :return: debiased特征 [C, H, W]
        """
        # 分辨率改变则重新生成基底
        if (self._base_B is None) or (self._cached_h_patch != h_patch) or (self._cached_w_patch != w_patch):
            self._build_fixed_basis(h_patch, w_patch, orig_w, orig_h)
        B_tensor = self._base_B
        C, H, W = orig_features.shape
        F_orig = orig_features
        F_orig_norm = F_orig / (torch.norm(F_orig, dim=0, keepdim=True) + 1e-6)
        F_flat = F_orig_norm.view(C, -1)
        F_proj = torch.matmul(B_tensor.T, F_flat)
        F_pos = torch.matmul(B_tensor, F_proj)
        F_debias_flat = F_flat - F_pos
        F_debias = F_debias_flat.view(C, H, W)
        return F_debias

# -------------------全局变量------------------
select_point = None
ref_feature = None
threshold = 50

def mouse_callback(event, x, y, flags, param):
    global select_point
    if event == cv2.EVENT_LBUTTONDOWN:
        select_point = (x, y)
        print(f"鼠标选取原图坐标 px={x}, py={y}")

if __name__ == '__main__':
    # KEEP_N在类初始化传入，统一管理
    dinov3_pytorch = DINOv3Pipeline(device="cuda", KEEP_N=20)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("摄像头打开失败")

    cv2.namedWindow("DINOv3-Live Match")
    cv2.setMouseCallback("DINOv3-Live Match", mouse_callback)
    cv2.createTrackbar("Sim Thresh(0-100)", "DINOv3-Live Match", threshold, 100, lambda x: None)
    patch_size = dinov3_pytorch.patch_size

   
    
    fps_display = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fps_start = time.time()
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        W_orig, H_orig = img_pil.size

        with torch.no_grad():
            feats, h_patch, w_patch, target_w, target_h = dinov3_pytorch.extract_features(img_pil)
            # 调用类内置debias，自动复用固定基底，无抖动
            feats_debias = dinov3_pytorch.debias(feats, h_patch, w_patch, W_orig, H_orig)
            C, fh, fw = feats_debias.shape

        # 修复后的坐标映射逻辑
        if select_point is not None and ref_feature is None:
            px, py = select_point
            scale_tw = target_w / W_orig
            scale_th = target_h / H_orig
            tx = px * scale_tw
            ty = py * scale_th
            fx = int(tx / patch_size)
            fy = int(ty / patch_size)
            fx = np.clip(fx, 0, fw - 1)
            fy = np.clip(fy, 0, fh - 1)
            ref_feature = feats_debias[:, fy, fx]
            print(f"映射后patch网格坐标 fy={fy}, fx={fx} | target图像素 tx={tx:.1f}, ty={ty:.1f}")

        heatmap = np.zeros((H_orig, W_orig), dtype=np.float32)
        if ref_feature is not None:
            current_thresh = cv2.getTrackbarPos("Sim Thresh(0-100)", "DINOv3-Live Match") / 100.0
            feat_flat = feats_debias.view(C, -1)
            ref_norm = ref_feature / (torch.norm(ref_feature) + 1e-6)
            feat_norm = feat_flat / (torch.norm(feat_flat, dim=0, keepdim=True) + 1e-6)
            sim_flat = torch.matmul(ref_norm, feat_norm)
            sim_2d = sim_flat.view(fh, fw)
            sim_np = sim_2d.cpu().numpy()
            heatmap = cv2.resize(sim_np, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)
            heatmap[heatmap < current_thresh] = 0.0

        heat_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        out_img = cv2.addWeighted(frame, 0.6, heat_color, 0.4, 0)

        #if select_point is not None:
        #    cv2.circle(out_img, select_point, 6, (0, 0, 255), -1)

        # ---- FPS 计算 ----
        now = time.time()
        fps_display = 1 / (now - fps_start)


        cv2.putText(out_img, f"FPS: {fps_display:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("DINOv3-Live Match", out_img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            select_point = None
            ref_feature = None
            print("C键：完全清空选点+参考特征，无热力图无红点")
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
