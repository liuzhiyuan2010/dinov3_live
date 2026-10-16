import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

# --- 配置 ---
PATCH_SIZE = 16
IMAGE_SIZE = 768
MASK_THRESHOLD = 128  # 用于二值化 mask，大于此值的像素视为白色（前景）


def image_transform(image, image_size=IMAGE_SIZE, patch_size=PATCH_SIZE):
    """
    图像处理：保持长宽比缩放，并确保尺寸是 patch_size 的倍数
    """
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    w, h = image.size
    # 计算缩放后的 patch 数量
    h_patches = int(image_size / patch_size)
    w_patches = int((w * image_size) / (h * patch_size))

    # 计算目标尺寸 (必须是 patch_size 的倍数)
    target_h = h_patches * patch_size
    target_w = w_patches * patch_size

    # Resize 并 ToTensor
    image_resized = F.to_tensor(F.resize(image, (target_h, target_w)))
    # Normalize
    image_resized_norm = F.normalize(image_resized, mean=IMAGENET_MEAN, std=IMAGENET_STD)

    return image_resized_norm, h_patches, w_patches


def get_mask_grid(mask_path, h_patches, w_patches, patch_size):
    """
    加载 mask 并调整大小以匹配特征网格
    返回二值化的 Tensor [1, h_patches, w_patches]
    """
    mask_img = Image.open(mask_path).convert("L")  # 转为灰度图

    # 关键：Mask 的缩放逻辑必须与 image_transform 完全一致
    # 计算目标尺寸
    target_h = h_patches * patch_size
    target_w = w_patches * patch_size

    # 使用 NEAREST 插值法调整 mask 大小，以保持边缘清晰，避免产生 0-255 之间的模糊值
    mask_resized = mask_img.resize((target_w, target_h), resample=Image.NEAREST)

    # 转为 Tensor [0, 1]
    mask_tensor = F.to_tensor(mask_resized)

    # 二值化：大于阈值的设为 1，否则为 0
    # 此时 mask_tensor 形状为 [1, H, W]
    mask_binary = (mask_tensor > (MASK_THRESHOLD / 255.0)).float()

    # 我们需要的是 patch 级别的 mask，所以需要 pooling 或者简单的 view 操作
    # 但这里为了简单且准确，我们直接把 mask 缩放到 patch 网格大小
    # 重新 resize 到 (w_patches, h_patches)
    mask_grid = mask_img.resize((w_patches, h_patches), resample=Image.NEAREST)
    mask_grid_tensor = F.to_tensor(mask_grid)
    mask_binary = (mask_grid_tensor > (MASK_THRESHOLD / 255.0)).float()

    return mask_binary


def get_image_features(image_path, mask_path=None):
    # 1. 加载原始图片
    raw_image = Image.open(image_path).convert("RGB")

    # 2. 预处理图片
    input_tensor, h_patches, w_patches = image_transform(raw_image)
    input_tensor = input_tensor.unsqueeze(0).to(device)  # [1, 3, H, W]

    print(f"✅ 图片加载完成: {image_path}")

    # 3. 执行前向传播
    with torch.inference_mode():
        # 提取特征
        feats = model.get_intermediate_layers(input_tensor, n=12, reshape=True, norm=True)
        dense_features = feats[-1].squeeze(0)  # [384, h, w]

    # 4. 处理 Mask 并提取前景特征
    if mask_path and os.path.exists(mask_path):
        # 获取对应的 mask 网格 [1, h, w]
        mask_grid = get_mask_grid(mask_path, h_patches, w_patches, PATCH_SIZE).to(device)

        # 将 mask 和特征都展平
        mask_flat = mask_grid.view(-1)  # [h*w]
        feat_flat = dense_features.view(384, -1)  # [384, h*w]

        # 提取 mask 为 1 的位置的特征
        # 逻辑索引：取出 feat_flat 中对应 mask_flat == 1 的列
        foreground_features = feat_flat[:, mask_flat == 1]  # [384, num_valid_patches]

        print(f"✅ 提取到前景 Patch 数量: {foreground_features.shape[1]}")

        if foreground_features.shape[1] == 0:
            print("⚠️ 警告：Mask 中没有检测到白色区域！请检查 mask 路径或颜色。")
            # 如果 mask 全黑，回退到全局平均，防止程序崩溃
            return dense_features.mean(dim=[1, 2]), raw_image, dense_features

        # 计算前景特征的平均向量
        ref_vector = foreground_features.mean(dim=1)  # [384]
        return ref_vector, raw_image, dense_features

    else:
        # 如果没有 mask，返回全局平均
        return dense_features.mean(dim=[1, 2]), raw_image, dense_features


# --- 设备与模型加载 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载模型 (请确保本地权重路径正确，且已解决联网问题)
try:
    # 如果你有本地源码，建议使用 sys.path.append 的方式加载，避免联网
    # 这里为了保持你原代码风格，暂保留 torch.hub，但请确保网络通畅或已配置好本地源

    # 1. 获取当前脚本的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 2. 拼接出本地 dinov3 源码的绝对路径
    # 这里的 'dinov3' 就是你刚才复制并重命名的文件夹名
    local_dinov3_path = os.path.join(current_dir, 'facebookresearch_dinov3_main')
    model = torch.hub.load(local_dinov3_path, 'dinov3_vits16', source='local', pretrained=False)
    local_weights_path = "./dinov3_wieghts/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    if os.path.exists(local_weights_path):
        state_dict = torch.load(local_weights_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval()
        print("✅ 本地权重加载成功！")
    else:
        print(f"❌ 找不到本地权重文件: {local_weights_path}")
        import sys;

        sys.exit()
except Exception as e:
    print(f"模型加载错误: {e}")
    import sys;

    sys.exit()

# --- 主流程 ---

# 1. 提取参考图特征 (传入 mask 路径)
# 注意：ref_feat 现在是一个 [384] 的向量，代表 mask 区域的平均特征
ref_vector, ref_img, ref_feat_map = get_image_features('dog1.jpg', 'dog1_mask.jpg')

# 2. 提取目标图特征 (不需要 mask)
tgt_vector, tgt_img, tgt_feat_map = get_image_features('people_dog.jpg', None)

# --- 相似度计算与可视化 ---

# 1. 准备目标图特征
h, w = tgt_feat_map.shape[1], tgt_feat_map.shape[2]
tgt_vectors = tgt_feat_map.view(384, -1).permute(1, 0)  # [H*W, 384]

# 2. 归一化
# ref_vector 已经是平均后的向量，需要重新归一化
ref_vector_norm = ref_vector / ref_vector.norm(p=2, dim=0, keepdim=True)
tgt_vectors_norm = tgt_vectors / tgt_vectors.norm(p=2, dim=1, keepdim=True)

# 3. 计算相似度
similarity_map = torch.matmul(tgt_vectors_norm, ref_vector_norm).squeeze()  # [H*W]
similarity_grid = similarity_map.view(h, w).cpu().numpy()

# 4. 可视化
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(ref_img)
plt.title("Reference Image (Masked)")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(tgt_img)
plt.title("Target Image")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(tgt_img)
# Resize 热力图以匹配原图大小
target_h, target_w = tgt_img.size[1], tgt_img.size[0]
heatmap = cv2.resize(similarity_grid, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

# 过滤低相似度 (可选)
filtered_heatmap = heatmap.copy()
filtered_heatmap[heatmap < 0.5] = 0

plt.imshow(filtered_heatmap, alpha=0.6, cmap='jet')
plt.title(f"Similarity Heatmap (Max: {similarity_grid.max():.2f})")
plt.axis('off')

plt.tight_layout()
plt.show()

# 5. 打印最相似的区域坐标
max_idx = np.unravel_index(np.argmax(similarity_grid), similarity_grid.shape)
print(f"最相似的区域位于网格坐标: {max_idx} (行, 列)")