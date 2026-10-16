import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os


PATCH_SIZE = 16
IMAGE_SIZE = 768



def image_transform(
    mask_image: Image,
    image_size: int = 768,
    patch_size: int = 16) -> torch.Tensor:

        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        w, h = mask_image.size
        h_patches = int(image_size / patch_size)
        w_patches = int((w * image_size) / (h * patch_size))
        image_resized =  F.to_tensor(F.resize(mask_image, (h_patches * patch_size, w_patches * patch_size)))
        image_resized_norm = F.normalize(image_resized, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        return image_resized_norm








# 1. 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

current_dir = os.path.dirname(os.path.abspath(__file__))
# 2. 拼接出本地 dinov3 源码的绝对路径
# 这里的 'dinov3' 就是你刚才复制并重命名的文件夹名
local_dinov3_path = os.path.join(current_dir, 'facebookresearch_dinov3_main')
model = torch.hub.load(local_dinov3_path, 'dinov3_vits16', source='local', pretrained=False)

# 加载本地权重
local_weights_path = "./dinov3_wieghts/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
state_dict = torch.load(local_weights_path, map_location=device, weights_only=False)
model.load_state_dict(state_dict, strict=True)
model.to(device)
model.eval()
print("✅ 本地权重加载成功！")






def get_image_features(image_path):
    raw_image = Image.open(image_path).convert("RGB")
    input_tensor = image_transform(raw_image).unsqueeze(0).to(device)
    print(f"✅ 图片加载完成。")



    # 5. 执行前向传播
    print("正在提取特征...")
    with torch.inference_mode():
        # 提取最后一层的密集特征，并自动重塑和归一化
        # n=12 表示提取所有12层的输出，我们取最后一层
        feats = model.get_intermediate_layers(input_tensor, n=12, reshape=True, norm=True)
        
        # feats 是一个列表，feats[-1] 是最后一层的输出
        # 形状为 [Batch, Dimension, Height, Width] -> [1, 384, w, h]
        #print(f'len of feats:{len(feats)}')
        dense_features = feats[-1]
        
        
        # 去掉 batch 维度 -> [384, w, h]
        dense_features = dense_features.squeeze(0)
        print(f'featshape:{dense_features.shape}')

    return dense_features,raw_image

ref_feat ,ref_img= get_image_features('dog1.jpg')

tgt_feat,tgt_img = get_image_features('people_dog.jpg')



# 2. 计算参考图的全局平均向量

ref_vector = ref_feat.mean(dim=[1, 2]) # 对 H, W 维度取平均
print(f"参考向量形状: {ref_vector.shape}") # [384]

# 3. 准备目标图特征进行对比
# tgt_feat 形状: [384, 14, 14]
# 将其变形为 [196, 384]，每一行代表一个 Patch 的特征

tgt_vectors = tgt_feat.view(384, -1).permute(1, 0) # [196, 384]

# 4. 计算余弦相似度
# 公式：(A · B) / (||A|| * ||B||)
# 由于 DINOv3 输出已经归一化 (norm=True)，点积近似等于余弦相似度
# 但为了严谨，我们再次标准化一下参考向量（因为平均操作破坏了归一化性质）

ref_vector_norm = ref_vector / ref_vector.norm(p=2, dim=0, keepdim=True)

tgt_vectors_norm = tgt_vectors / tgt_vectors.norm(p=2, dim=1, keepdim=True) # [196, 384]

# 计算点积 (矩阵乘法)
# [196, 384] @ [384, 1] -> [196, 1]
similarity_map = torch.matmul(tgt_vectors_norm, ref_vector_norm.T).squeeze()


h, w = tgt_feat.shape[1], tgt_feat.shape[2]
similarity_grid = similarity_map.view(h, w).cpu().numpy()

print(f"相似度最大值: {similarity_grid.max():.4f}")
print(f"相似度最小值: {similarity_grid.min():.4f}")

# 5. 可视化结果
plt.figure(figsize=(15, 5))

# 图1: 参考图
plt.subplot(1, 3, 1)
plt.imshow(ref_img)
plt.title("Reference Image (dog1.jpg)")
plt.axis('off')

# 图2: 目标图
plt.subplot(1, 3, 2)
plt.imshow(tgt_img)
plt.title("Target Image (people_dog.jpg)")
plt.axis('off')

# 图3: 相似度热力图
plt.subplot(1, 3, 3)
# 将目标图作为背景
plt.imshow(tgt_img)
# 叠加热力图，使用 'jet' 或 'hot' 颜色映射，alpha控制透明度
# 注意：需要将 similarity_grid resize 到原图大小才能完美对齐，或者直接画在 patch 网格上
# 这里为了简单，我们使用 nearest 插值放大到原图大小
target_h, target_w = tgt_img.size[1], tgt_img.size[0]

filtered_grid = similarity_grid.copy()

filtered_grid[similarity_grid < 0.45] = 0

heatmap = cv2.resize(filtered_grid, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

plt.imshow(heatmap, alpha=0.5, cmap='jet') 
plt.title(f"Similarity Heatmap (Max: {similarity_grid.max():.2f})")
plt.axis('off')

plt.tight_layout()
plt.show()

# 6. (可选) 打印最相似的区域坐标
max_idx = np.unravel_index(np.argmax(similarity_grid), similarity_grid.shape)
print(f"最相似的区域位于网格坐标: {max_idx} (行, 列)")

