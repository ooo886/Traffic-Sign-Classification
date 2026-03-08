"""
1_preprocess_and_save.py
功能：预处理所有图片并保存为Tensor文件
"""

from torchvision import transforms  # 提供图像预处理功能
from torch.utils.data import DataLoader, Dataset  # PyTorch数据加载工具
from PIL import Image  # Python图像处理库
import os  # 文件和路径操作
import torch  # PyTorch主库
import numpy as np  # 数值计算
import matplotlib.pyplot as plt  # 绘图工具

# ========== 1. 定义预处理 ==========
# 训练集增强并归一化
train_transform = transforms.Compose([
    transforms.RandomRotation(degrees=5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.Resize((224,224)),
    transforms.ToTensor(),# 将图像格式转化为pytorch的张量格式
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #再进行数据标准化
])

# 验证/测试集预处理
#重复上述操作，设置大小224×224，以及转化成张量格式，再标准化
val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ========== 2. 定义数据集类 ==========
class TrafficSignDataset(Dataset):
    # 初始化：扫描文件夹，记录所有图片路径
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir   # 保存根目录
        self.transform = transform # 保存预处理方法
        self.images = []  # 存储图片路径
        self.labels = []  # 存储标签
        self.classes = [] # 存储类别名

        # 获取所有类别
        self.classes = sorted([d for d in os.listdir(root_dir)
        #os.listdir(root_dir)列出root_dir目录下的所有文件和文件夹名称，并转化为列表
        # for d in os.listdir(root_dir)，d代表每个文件的名称
                               if os.path.isdir(os.path.join(root_dir, d))]) #过滤条件：只保留目录
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        #创建类别名到数字的映射

        print(f"📁 加载 {os.path.basename(root_dir)} 数据集...")
        # 收集所有图片路径
        for class_name in self.classes: #遍历所有文件夹
            class_path = os.path.join(root_dir, class_name) #合成最后的途径
            for img_name in os.listdir(class_path):  #在最里面的文件夹里遍历
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')): #如果格式是.jpg则继续循环
                    self.images.append(os.path.join(class_path, img_name))  #并且存入图片的路径
                    self.labels.append(self.class_to_idx[class_name])  #也存入图片的编号

        print(f"   ├─ 类别数: {len(self.classes)}")
        print(f"   └─ 图片数: {len(self.images)}")

    def __len__(self):
        return len(self.images) # 返回数据集大小

    def __getitem__(self, idx):
        # 根据索引获取一张图片
        img_path = self.images[idx]  # 获取图片路径
        image = Image.open(img_path).convert('RGB')  # 读取图片
        label = self.labels[idx] # 获取标签

        if self.transform:
            image = self.transform(image) # 应用预处理

        return image, label # 返回处理好的图片和标签

# ========== 3. 加载并预处理所有数据 ==========
print("=" * 60)
print("📊 第一阶段：数据预处理")
print("=" * 60)

base_path = r"C:\Users\Administrator\Desktop\traffic-sign-dataset-classification\traffic_Data"
# 获取所有的路径
train_path = os.path.join(base_path, "DATA") #训练集路径
val_path = os.path.join(base_path, "VAL")   #验证集集路径
test_path = os.path.join(base_path, "TEST")  #测试集路径

# 创建保存目录，用于保存预处理后的图像数据
save_dir = 'saved_data'
os.makedirs(save_dir, exist_ok=True)

# ===== 处理训练集 =====
print("\n🔄 正在预处理训练集...")
train_dataset = TrafficSignDataset(root_dir=train_path, transform=train_transform)

# 预处理所有训练集图片并保存
train_images = []
train_labels = []

for i in range(len(train_dataset)):
    if i % 500 == 0:
        print(f"   进度: {i}/{len(train_dataset)}")
    img, label = train_dataset[i]
    train_images.append(img.cpu()) #将图片保存至cpu上
    train_labels.append(label)

# 保存为Tensor，保存形式为saved_data
torch.save({
    'images': torch.stack(train_images),  # 堆叠成 [N, C, H, W]
    'labels': torch.tensor(train_labels), # 转为Tensor
    'classes': train_dataset.classes      # 保存类别名
}, os.path.join(save_dir, 'train_data.pt'))

print(f"✅ 训练集已保存: {len(train_images)} 张图片")

# ===== 处理验证集 =====
print("\n🔄 正在预处理验证集...")
val_dataset = TrafficSignDataset(root_dir=val_path, transform=val_test_transform)

val_images = []
val_labels = []

for i in range(len(val_dataset)):
    img, label = val_dataset[i]
    val_images.append(img.cpu())
    val_labels.append(label)

torch.save({
    'images': torch.stack(val_images),
    'labels': torch.tensor(val_labels),
    'classes': val_dataset.classes
}, os.path.join(save_dir, 'val_data.pt'))

print(f"✅ 验证集已保存: {len(val_images)} 张图片")

# ===== 处理测试集 =====
print("\n🔄 正在预处理测试集...")
test_dataset = TrafficSignDataset(root_dir=test_path, transform=val_test_transform)

test_images = []
test_labels = []

for i in range(len(test_dataset)):
    img, label = test_dataset[i]
    test_images.append(img.cpu())
    test_labels.append(label)

torch.save({
    'images': torch.stack(test_images),
    'labels': torch.tensor(test_labels),
    'classes': test_dataset.classes
}, os.path.join(save_dir, 'test_data.pt'))

print(f"✅ 测试集已保存: {len(test_images)} 张图片")

# ========== 4. 可视化增强效果 ==========
def show_augmented_samples():
    """显示数据增强后的样本"""
    fig, axes = plt.subplots(1, 5, figsize=(15, 3)) # 创建5个子图

    def denormalize(tensor):
        # 反标准化：将预处理后的Tensor转回可视化格式
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = tensor * std + mean
        return tensor.clamp(0, 1)  # 限制在0-1范围内

    for i in range(5):
        idx = np.random.randint(len(train_images)) # 随机选一张
        img = train_images[idx] # 获取图片
        label = train_labels[idx] # 获取标签

        img = denormalize(img) # 反标准化
        img = img.permute(1, 2, 0).numpy() # 调整维度顺序 [C,H,W] → [H,W,C]

        axes[i].imshow(img) # 显示图片
        axes[i].set_title(f'{train_dataset.classes[label]}') # 设置标题
        axes[i].axis('off')  # 关闭坐标轴

    plt.tight_layout()
    plt.savefig('results/augmentation_samples.png', dpi=150, bbox_inches='tight') # 保存图片
    plt.show() # 显示图片

os.makedirs('results', exist_ok=True)
print("\n📸 数据增强效果示例:")
show_augmented_samples()

print("\n" + "=" * 60)
print("✅ 数据预处理完成！所有数据已保存到 saved_data/ 文件夹")
print("   下一步：运行 2_train_resnet18.py 开始训练")