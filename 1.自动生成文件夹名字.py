import os


def create_numbered_folders(base_path, start=0, end=57):
    """
    在指定路径下创建从start到end的编号文件夹

    Args:
        base_path: 要创建文件夹的路径
        start: 起始编号
        end: 结束编号
    """
    # 确保基础路径存在
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        print(f"📁 创建基础路径: {base_path}")

    print(f"开始创建文件夹 {start} 到 {end}...")
    print("-" * 40)

    for i in range(start, end + 1):
        folder_name = str(i)  # 直接使用数字作为文件夹名
        folder_path = os.path.join(base_path, folder_name)

        try:
            os.makedirs(folder_path, exist_ok=True)
            print(f"✅ 创建: {folder_name}/")
        except Exception as e:
            print(f"❌ 创建失败 {folder_name}: {e}")

    print("-" * 40)
    print(f"🎉 完成！共创建 {end - start + 1} 个文件夹")


# 使用示例
base_path = r"C:\Users\Administrator\Desktop\traffic-sign-dataset-classification\traffic_Data\VAL"
create_numbered_folders(base_path, 0, 57)