import os
from collections import defaultdict


def count_images(folder_path, folder_name=""):
    """
    统计文件夹中的图片数量

    Args:
        folder_path: 要统计的文件夹路径
        folder_name: 文件夹的显示名称
    """
    # 常见的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}

    # 检查路径是否存在
    if not os.path.exists(folder_path):
        print(f"❌ 路径不存在: {folder_path}")
        return {}, 0, 0

    # 用于存储统计结果
    stats = {}
    total_images = 0
    class_count = 0

    # 显示当前统计的文件夹
    if folder_name:
        print(f"\n📂 === 正在统计 {folder_name} 文件夹 ===")

    # 遍历文件夹
    for root, dirs, files in os.walk(folder_path):
        # 获取当前文件夹名称
        current_folder = os.path.basename(root)

        # 如果是根目录，跳过（避免把根目录算作类别）
        if root == folder_path:
            continue

        # 统计当前文件夹的图片
        image_count = 0
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_count += 1

        if image_count > 0:  # 只显示有图片的文件夹
            stats[current_folder] = image_count
            total_images += image_count
            class_count += 1
            print(f"  📁 {current_folder}: {image_count:4d} 张图片")

    # 如果没有找到子文件夹，检查根目录是否有图片
    if class_count == 0:
        # 直接统计根目录的图片
        root_files = os.listdir(folder_path)
        image_count = 0
        for file in root_files:
            if os.path.isfile(os.path.join(folder_path, file)) and \
                    any(file.lower().endswith(ext) for ext in image_extensions):
                image_count += 1

        if image_count > 0:
            total_images = image_count
            class_count = 1
            print(f"  📁 (根目录): {image_count:4d} 张图片")
            stats['(根目录)'] = image_count
        else:
            print("  ⚠️ 没有找到图片")

    print(f"\n  📊 总计: {total_images:4d} 张图片 (来自 {class_count} 个类别)")

    # 返回详细统计
    return stats, total_images, class_count


# 设置路径
base_path = r"C:\Users\Administrator\Desktop\traffic-sign-dataset-classification\traffic_Data"
data_path = os.path.join(base_path, "DATA")
test_path = os.path.join(base_path, "TEST")
val_path = os.path.join(base_path, "VAL")

print("=" * 60)
print("📸 交通标志数据集统计")
print("=" * 60)

# 统计 DATA 文件夹
data_stats, data_total, data_classes = count_images(data_path, "DATA")

# 统计 TEST 文件夹
test_stats, test_total, test_classes = count_images(test_path, "TEST")

# 统计 VAL 文件夹（如果存在）
val_stats, val_total, val_classes = count_images(val_path, "VAL")

# 打印汇总信息
print("\n" + "=" * 60)
print("📊 数据集汇总")
print("=" * 60)
print(f"{'文件夹':<15} {'图片数量':<15} {'类别数量':<15}")
print("-" * 60)
print(f"{'DATA':<15} {data_total:<15} {data_classes:<15}")
print(f"{'TEST':<15} {test_total:<15} {test_classes:<15}")
print(f"{'VAL':<15} {val_total:<15} {val_classes:<15}")
print("-" * 60)
print(f"{'总计':<15} {data_total + test_total + val_total:<15} {'-':<15}")

# 可选：显示每个文件夹的详细类别信息
show_details = input("\n是否显示每个文件夹的详细类别统计？(y/n): ")
if show_details.lower() == 'y':
    print("\n" + "=" * 60)
    print("📋 详细类别统计")
    print("=" * 60)

    if data_stats:
        print("\n📁 DATA 文件夹的类别分布:")
        for class_name, count in sorted(data_stats.items()):
            print(f"  {class_name}: {count}张")

    if test_stats:
        print("\n📁 TEST 文件夹的类别分布:")
        for class_name, count in sorted(test_stats.items()):
            print(f"  {class_name}: {count}张")

    if val_stats:
        print("\n📁 VAL 文件夹的类别分布:")
        for class_name, count in sorted(val_stats.items()):
            print(f"  {class_name}: {count}张")

print("\n" + "=" * 60)
print("✅ 统计完成！")
print("=" * 60)