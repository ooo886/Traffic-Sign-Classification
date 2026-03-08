import os
import shutil
import re


def move_val_images():
    """
    将VAL根目录的图片移动到对应的0-57文件夹
    图片名格式: 000_xxx.jpg -> 移动到 0/ 文件夹
    """
    base_path = r"C:\Users\Administrator\Desktop\traffic-sign-dataset-classification\traffic_Data"
    val_path = os.path.join(base_path, "VAL")

    print("=" * 60)
    print("📁 开始整理VAL文件夹")
    print("=" * 60)

    # 检查VAL文件夹是否存在
    if not os.path.exists(val_path):
        print(f"❌ VAL文件夹不存在: {val_path}")
        return

    # 获取VAL根目录的所有图片
    all_files = [f for f in os.listdir(val_path)
                 if os.path.isfile(os.path.join(val_path, f))]

    image_files = [f for f in all_files
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    print(f"\n📊 在VAL根目录找到 {len(image_files)} 张图片")

    if not image_files:
        print("✅ 根目录没有图片，无需移动")
        return

    # 统计信息
    moved_count = 0
    unknown_files = []
    class_stats = {str(i): 0 for i in range(58)}

    print("\n🔄 开始移动图片...")
    print("-" * 40)

    for filename in image_files:
        # 提取前缀（000, 001, ...）
        match = re.match(r'^(\d{3})_', filename)

        if match:
            prefix = match.group(1)  # 如 "000"
            class_num = str(int(prefix))  # 转为整数再转回字符串，去掉前导0

            # 检查是否在0-57范围内
            if 0 <= int(class_num) <= 57:
                src = os.path.join(val_path, filename)
                dst_dir = os.path.join(val_path, class_num)

                # 确保目标文件夹存在
                if os.path.exists(dst_dir):
                    dst = os.path.join(dst_dir, filename)
                    shutil.move(src, dst)
                    moved_count += 1
                    class_stats[class_num] += 1

                    # 显示进度（每10张显示一次）
                    if moved_count % 10 == 0 or moved_count <= 10:
                        print(f"  ✅ {filename} -> {class_num}/")
                else:
                    print(f"  ⚠️ 目标文件夹不存在: {class_num}/")
                    unknown_files.append(filename)
            else:
                print(f"  ⚠️ 编号超出范围 (0-57): {prefix}")
                unknown_files.append(filename)
        else:
            unknown_files.append(filename)

    print("-" * 40)
    print(f"\n📊 移动完成！")
    print(f"   ✅ 成功移动: {moved_count} 张图片")
    print(f"   ❓ 无法移动: {len(unknown_files)} 张图片")

    # 显示各类别移动数量
    print(f"\n📈 各类别移动情况:")
    classes_with_images = [(k, v) for k, v in class_stats.items() if v > 0]
    for class_num, count in sorted(classes_with_images):
        print(f"   类别 {class_num}: {count:3d} 张")

    # 显示无法移动的文件
    if unknown_files:
        print(f"\n⚠️ 无法移动的文件 (前10个):")
        for f in unknown_files[:10]:
            print(f"   ❓ {f}")
        if len(unknown_files) > 10:
            print(f"   ... 等共 {len(unknown_files)} 张")

    print("\n" + "=" * 60)
    print("✅ VAL文件夹整理完成！")
    print("=" * 60)


# 运行
move_val_images()