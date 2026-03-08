"""
3_load_and_test_model.py
功能：加载训练好的模型并进行测试
"""
import torch
import torchvision.models as models
import torch.nn as nn
import os
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np

# ========== 1. 配置 ==========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# ========== 2. 加载类别信息 ==========
# 从之前保存的数据中获取类别名称
if os.path.exists('saved_data/train_data.pt'):
    train_data = torch.load('saved_data/train_data.pt')
    class_names = train_data['classes']
    num_classes = len(class_names)
    print(f"类别数: {num_classes}")
    print(f"类别名称: {class_names}")
else:
    # 如果找不到，手动设置（根据您的实际情况修改）
    class_names = ['class_0', 'class_1', 'class_2', 'class_3', 'class_4',
                   'class_5', 'class_6', 'class_7', 'class_8', 'class_9']
    num_classes = 10

# ========== 3. 创建相同的模型结构 ==========
model = models.resnet18(weights=None) #创建一个空的ResNet-18模型骨架
num_features = model.fc.in_features #获取输入的特征数
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.BatchNorm1d(256),
    nn.Dropout(0.3),
    nn.Linear(256, num_classes)  # 使用实际的类别数
)

# ========== 4. 加载模型权重 ==========
model_path = 'checkpoints/resnet18_final.pth'  # 选择要加载的文件

if not os.path.exists(model_path):
    print(f"错误: 找不到模型文件 {model_path}")
    print("可用的模型文件:")
    for f in os.listdir('checkpoints'):
        print(f"  - {f}")
    exit()

print(f"\n加载模型: {model_path}")
checkpoint = torch.load(model_path, map_location=device)

# 查看保存的内容，checkpoint是字典结构
print("\n文件内容:")
if isinstance(checkpoint, dict):
    for key in checkpoint.keys():
        if key == 'val_acc':
            print(f"  - 验证准确率: {checkpoint[key]}%")
        elif key == 'epoch':
            print(f"  - 训练轮数: {checkpoint[key]}")
        elif key == 'class_names':
            print(f"  - 类别名称: {checkpoint[key]}")
        else:
            print(f"  - {key}")

# 加载权重
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint: #确定是否是字典格式
    model.load_state_dict(checkpoint['model_state_dict']) #从字典中提取权重，再加载权重出来
    best_acc = checkpoint.get('val_acc', 'N/A') #安全地获取值，如果不存在就返回默认值'N/A'
    epoch = checkpoint.get('epoch', 'N/A') #安全地获取值，如果不存在就返回默认值'N/A'
    print(f"\n✅ 模型加载成功！")
    print(f"   最佳准确率: {best_acc}%")
    print(f"   训练轮数: {epoch}")
else:
    model.load_state_dict(checkpoint)
    print("\n✅ 模型权重加载成功！")

# ========== 5. 设置为评估模式 ==========
model = model.to(device)  #移送到cuda上
model.eval()
print("模型已设置为评估模式")


# ========== 6. 简单测试函数 ==========
def test_single_image(model, image_tensor, class_names):
    """
    测试单张图片
    image_tensor: 预处理好的图片张量 [C, H, W]
    """
    model.eval()
    with torch.no_grad():
        # 添加batch维度并移动到设备
        image = image_tensor.unsqueeze(0).to(device)

        # 预测
        outputs = model(image)  #输入一张图片，通过模型得到输出
        probabilities = torch.nn.functional.softmax(outputs, dim=1) #输出经过softmax函数的归一化得到概率
        predicted_class = torch.argmax(probabilities, dim=1).item() #找到概率最大的索引
        confidence = probabilities[0][predicted_class].item()

    return class_names[predicted_class], confidence


print("\n🎯 模型准备就绪！可以使用 test_single_image() 函数进行预测")
print("示例: class_name, confidence = test_single_image(model, your_image_tensor, class_names)")

# ========== 7. 如果有测试集，可以评估整体性能 ==========
if os.path.exists('saved_data/test_data.pt'):
    print("\n📊 加载测试集进行评估...")
    test_data = torch.load('saved_data/test_data.pt')
    test_images = test_data['images']
    test_labels = test_data['labels']

    # 创建数据加载器
    test_dataset = torch.utils.data.TensorDataset(test_images, test_labels)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 评估并收集所有预测结果
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            # 统计整体准确率
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 收集所有预测结果用于详细分析
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = 100. * correct / total
    print(f"✅ 测试集准确率: {test_acc:.2f}%")

# ========== 8. 生成完整的分类报告 ==========
print("\n" + "=" * 50)
print("📊 生成完整的分类报告")
print("=" * 50)

# 方法2.1：使用sklearn的分类报告
print("\n1️⃣ sklearn分类报告:")
print("-" * 60)
# 算出精确率、召回率、F1分数、支持度（样本数）
report = classification_report(all_labels, all_preds,
                               target_names=class_names,
                               digits=4)
print(report)

# 方法2.2：创建详细的数据表格
print("\n2️⃣ 详细数据表格:")
print("-" * 60)

# 计算每个类别的指标
class_metrics = []
for i in range(num_classes):
    # 找出当前类别的所有样本
    true_positive = sum([1 for p, l in zip(all_preds, all_labels) if p == i and l == i]) #TP
    false_positive = sum([1 for p, l in zip(all_preds, all_labels) if p == i and l != i]) #FP
    false_negative = sum([1 for p, l in zip(all_preds, all_labels) if p != i and l == i]) #FN
    true_negative = sum([1 for p, l in zip(all_preds, all_labels) if p != i and l != i]) #TN

    # 直接计算总数和正确数
    total = sum([1 for l in all_labels if l == i])  # 类别i的总样本数
    correct = sum([1 for p, l in zip(all_preds, all_labels) if p == l and l == i])  # 类别i的正确预测数

    accuracy = 100. * correct / total if total > 0 else 0
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    class_metrics.append({
        '类别': class_names[i],
        '准确率(%)': accuracy,
        '精确率': precision,
        '召回率': recall,
        'F1分数': f1,
        '正确数': correct,
        '总数': total
    })

# 创建DataFrame并打印
df = pd.DataFrame(class_metrics)
print(df.to_string(index=False))

# 保存到CSV文件
df.to_csv('results/class_metrics.csv', index=False, encoding='utf-8-sig')
print("\n✅ 详细指标已保存到: results/class_metrics.csv")

# 方法2.3：找出表现最好和最差的类别
print("\n3️⃣ 表现分析:")
print("-" * 60)

# 按准确率排序
df_sorted = df.sort_values('准确率(%)', ascending=False)

print("🎯 表现最好的3个类别:")
for i in range(min(3, len(df_sorted))):
    row = df_sorted.iloc[i]
    print(f"   {i + 1}. {row['类别']}: {row['准确率(%)']:.2f}% ({int(row['正确数'])}/{int(row['总数'])})")

print("\n⚠️ 表现最差的3个类别:")
for i in range(min(3, len(df_sorted))):
    row = df_sorted.iloc[-i - 1]
    print(f"   {i + 1}. {row['类别']}: {row['准确率(%)']:.2f}% ({int(row['正确数'])}/{int(row['总数'])})")