"""
2_train_resnet18.py
功能：加载预处理好的数据，训练ResNet-18模型
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import os
from datetime import datetime
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from collections import Counter
import numpy as np

# 创建results目录保存结果，如果目录已存在不会报错
os.makedirs('results', exist_ok=True)

# ========== 1. 加载预处理好的数据 ==========
print("=" * 60)
print("📊 加载预处理好的数据")
print("=" * 60)

save_dir = 'saved_data'

# 检查'saved_data'文件是否存在
if not os.path.exists(save_dir):
    print(f"❌ 错误: {save_dir} 目录不存在！")
    print("   请先运行 1.基础预处理和验证预处理效果.py")
    exit()

# 加载训练集，train_data是字典格式，键为images（张量）、labels（张量）、classes（列表）
train_data = torch.load(os.path.join(save_dir, 'train_data.pt')) #加载之前保存的PyTorch张量文件
train_images = train_data['images'] #读取图像张量，形状为 [N, C, H, W]
train_labels = train_data['labels'] #读取标签张量，形状为 [N]
class_names = train_data['classes'] #读取类别名称列表

# 加载验证集
# val_images 存图片，val_labels 存每张图片对应的正确答案
val_data = torch.load(os.path.join(save_dir, 'val_data.pt'))
val_images = val_data['images'] #读取图像张量，形状为 [N, C, H, W]
val_labels = val_data['labels'] #读取标签张量，形状为 [N]

# 加载测试集
test_data = torch.load(os.path.join(save_dir, 'test_data.pt'))
test_images = test_data['images'] #读取图片张量，形状为 [N, C, H, W]
test_labels = test_data['labels'] #读取标签张量，形状为 [N]

print(f"\n✅ 数据加载成功!")
print(f"   训练集: {len(train_images)} 张")
print(f"   验证集: {len(val_images)} 张")
print(f"   测试集: {len(test_images)} 张")
print(f"   类别数: {len(class_names)}")

# ========== 2. 创建数据加载器DataLoader ==========
batch_size = 32 # 批次大小，即一次喂给模型32张图片，之后更新参数

# 创建TensorDataset
#train_dataset = torch.utils.data.TensorDataset(train_images, train_labels)

# 训练时的在线增强（轻量级）
# 在训练的时候数据量不会增加，但是能让模型看到更多样的数据
# train_transform = transforms.Compose([
#     transforms.RandomRotation(degrees=5),  # 在±5度内随机旋转
#     transforms.ColorJitter(brightness=0.1, contrast=0.1), # 亮度随机变化±10% 对比度随机变化±10%
# ])

# 训练集、验证集和测试集不需要归一化
train_transform = None
val_transform = None
# 自定义一个支持数据增强的数据集类
class AugmentedDataset(torch.utils.data.TensorDataset):
    """
    继承自TensorDataset，添加了数据增强功能
    """
    def __init__(self, images, labels, transform=None):
        super().__init__(images, labels) # 这时 img 是原始图片，还没增强
        self.transform = transform

    def __getitem__(self, index): #定义获取图片
        img, label = super().__getitem__(index)
        # 第2步：如果有定义增强方法，就应用增强
        if self.transform:
            img = self.transform(img)
        # 第3步：返回处理后的图片和标签
        return img, label

# 将训练集的图像和标签打包，并且额外做一些数据增强
train_dataset = AugmentedDataset(train_images, train_labels, transform=train_transform)
# 将验证集图像和标签配对打包
val_dataset = AugmentedDataset(val_images, val_labels, transform=None)
# 将测试集图像和标签配对打包
test_dataset = AugmentedDataset(test_images, test_labels, transform=None)

# 创建DataLoader
# train_dataset是要加载的数据集，batch_size=batch_size是指批次大小为32（即每次取32张）
#shuffle=True是指打乱数据，num_workers=0是指不用多进程加载数据
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
)

print(f"\n📊 数据加载器创建完成:")
print(f"   训练集批次: {len(train_loader)}")
print(f"   验证集批次: {len(val_loader)}")
print(f"   测试集批次: {len(test_loader)}")

# ========== 3. 设置训练参数 ==========
class Config:
    num_classes = len(class_names)  # 类别数（从数据中获取）
    batch_size = batch_size  # 批次大小（32）
    num_epochs = 100  #训练100轮
    learning_rate = 0.001  #学习率设置为0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #device自动选择gpu
    save_dir = 'checkpoints' #将训练的最优模型参数保存至checkpoints
    os.makedirs(save_dir, exist_ok=True) #并且创建这个文件

config = Config() #创建这样的实例
print(f"\n⚙️ 训练配置:")
print(f"   设备: {config.device}")
print(f"   批次大小: {config.batch_size}")
print(f"   训练轮数: {config.num_epochs}")
print(f"   学习率: {config.learning_rate}")

# ========== 4. 构建 ResNet-18 模型 ==========
print("\n🏗️ 构建 ResNet-18 模型...")

# 加载预训练模型
model = models.resnet18(weights='IMAGENET1K_V1') # 加载在ImageNet数据集上预训练的权重
# 获取原模型最后一层的输入特征数
num_features = model.fc.in_features
#替换最后一层，适配新的训练任务
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.BatchNorm1d(256),
    nn.Dropout(0.3),
    nn.Linear(256, config.num_classes)
)
# 整个模型一起移到GPU
model = model.to(config.device)
#model.fc = nn.Linear(num_features, config.num_classes)

# 冻结策略：解冻最后两层（layer4和fc层）
print("\n🔧 冻结策略: 解冻最后两层 (layer4和fc层)")
for name, param in model.named_parameters():
    # 解冻layer4（最后一个卷积块）和fc层
    if 'layer4' in name or 'fc' in name:
        param.requires_grad = True
        print(f"   ✅ 解冻: {name}")
    else:
        param.requires_grad = False

# 添加参数统计
# all_params = model.parameters()  返回模型的所有参数
# 只保留 requires_grad = True 的参数
# 第3步：计算总元素个数
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"\n📊 参数统计:")
print(f"   总参数: {total_params:,}")
print(f"   可训练参数: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")

# ========== 5. 定义损失和优化器 ==========
# 先计算类别权重（在数据加载后添加）
class_counts = Counter(train_labels.numpy())#将pytorch张量全部转化为numpy，并统计每个字出现的次数

# 检查类别分布
print("类别分布:")
for i in range(len(class_names)):
    count = class_counts.get(i, 0)
    print(f"  类别 {i}: {count} 张")

# 如果某个类别数量为0，权重会无穷大
# 添加安全检查

class_weights = torch.tensor([
    len(train_labels) / (len(class_names) * max(class_counts.get(i, 1), 1))
    for i in range(len(class_names))
]).float().to(config.device)

# 如果某个类别真的为0，给它一个合理的权重
if 0 in class_counts.values():
    print("⚠️ 警告：存在空类别，已自动调整权重")
    class_weights = class_weights / class_weights.min()  # 归一化

# 然后使用带权重的损失函数
criterion = nn.CrossEntropyLoss(weight=class_weights)  # ✅ 使用类别权重

# 为不同层设置不同的学习率
optimizer = optim.Adam([
    {'params': model.layer4.parameters(), 'lr': 0.0005},  # 较高学习率
    {'params': model.fc.parameters(), 'lr': 0.001},      # 最高学习率
    {'params': [p for n, p in model.named_parameters()
                if not any(x in n for x in ['layer4', 'fc'])],
     'lr': 0.0001},  # 冻结层用低学习率（如果有的话）
], lr=0.001, weight_decay=1e-4)

# 使用余弦退火，让学习率更平滑地下降
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100,  # 周期长度
    eta_min=1e-6  # 最小学习率
)
# ========== 6. 训练循环 ==========
print("\n🚀 开始训练...")
print("=" * 60)

best_val_acc = 0.0
train_losses = [] #训练集的误差
val_accuracies = []  #验证集的准确率

#开始100次的训练循环
for epoch in range(config.num_epochs):
    # 训练阶段
    model.train() #开始训练模型
    running_loss = 0.0 # 用来累加这一轮所有batch的损失值

    for batch_idx, (images, labels) in enumerate(train_loader): #遍历train_loader中的数据
        images, labels = images.to(config.device), labels.to(config.device) #将数据迁移至gpu

        optimizer.zero_grad() #清空上一次计算的梯度
        outputs = model(images) # 模型对这批图片进行预测
        loss = criterion(outputs, labels) # 比较模型的预测(outputs)和正确答案(labels)
        loss.backward()  # 计算每个参数的梯度
        # 添加梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()  # 根据计算好的梯度，更新模型参数

        running_loss += loss.item() # 记录这个batch的损失值

        # 打印进度（每10个batch）,每10个batch打印一次当前损失
        if (batch_idx + 1) % 10 == 0:
            print(f'   Batch [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}')

    # 计算这一轮的平均损失，记录下来用于画图
    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # 验证阶段
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(config.device), labels.to(config.device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_acc = 100. * correct / total
    val_accuracies.append(val_acc)

    # 更新学习率
    scheduler.step()

    print(f"\n📊 Epoch [{epoch+1}/{config.num_epochs}] 结果:")
    print(f"   平均损失: {avg_train_loss:.4f}")
    print(f"   验证准确率: {val_acc:.2f}%")
    print(f"   学习率: {optimizer.param_groups[0]['lr']:.6f}")

    # 保存最佳模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(config.save_dir, f'resnet18_best_{timestamp}.pth')

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'train_loss': avg_train_loss,
            'class_names': class_names
        }, save_path)

        print(f"   ✅ 保存最佳模型 (准确率: {val_acc:.2f}%) -> {save_path}")

# ========== 7. 训练结束 ==========
print("\n" + "=" * 60)
print("🎉 训练完成!")
print(f"   最佳验证准确率: {best_val_acc:.2f}%")

# 保存最终模型
final_path = os.path.join(config.save_dir, 'resnet18_final.pth')
torch.save(model.state_dict(), final_path)
print(f"   最终模型已保存: {final_path}")

# 绘制训练曲线
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses)
plt.title('ResNet-18 Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(val_accuracies)
plt.title('ResNet-18 Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.grid(True)

plt.tight_layout()
plt.savefig('results/training_curves.png', dpi=150)
plt.show()

print("\n📈 训练曲线已保存: results/training_curves.png")
print("\n✅ 所有任务完成！")
