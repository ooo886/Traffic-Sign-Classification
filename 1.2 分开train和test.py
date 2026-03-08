import os
import random
import shutil


def split_test_to_val_flat(test_path, val_path, split_ratio=0.5):
    """
    适用于TEST文件夹直接存放图片的情况
    """

    if not os.path.exists(test_path):
        print(f"❌ TEST文件夹不存在: {test_path}")
        return

    # 创建验证集文件夹
    os.makedirs(val_path, exist_ok=True)

    # 获取所有图片
    all_images = [f for f in os.listdir(test_path)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'))]

    if not all_images:
        print("❌ TEST文件夹中没有图片")
        return

    # 随机选择一半
    num_to_move = int(len(all_images) * split_ratio)
    images_to_move = random.sample(all_images, num_to_move)

    print("=" * 50)
    print("🔄 开始从TEST中抽取图片到验证集")
    print("=" * 50)
    print(f"📊 TEST中共有 {len(all_images)} 张图片")
    print(f"🎲 将移动 {num_to_move} 张到验证集")

    # 移动图片
    moved_count = 0
    for img in images_to_move:
        src = os.path.join(test_path, img)
        dst = os.path.join(val_path, img)
        shutil.move(src, dst)
        moved_count += 1
        print(f"  ✅ 移动: {img}")

    print("\n" + "=" * 50)
    print("✅ 抽取完成！")
    print(f"   ├─ 总共移动: {moved_count} 张图片到验证集")
    print(f"   ├─ 验证集路径: {val_path}")
    print(f"   └─ TEST剩余: {len(all_images) - moved_count} 张")
    print("=" * 50)


# 使用
test_path = r"C:\Users\Administrator\Desktop\traffic-sign-dataset-classification\traffic_Data\TEST"
val_path = r"C:\Users\Administrator\Desktop\traffic-sign-dataset-classification\traffic_Data\VAL"
split_test_to_val_flat(test_path, val_path, split_ratio=0.5)
