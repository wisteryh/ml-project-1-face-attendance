"""
机器学习项目一 - Step 2: 构建人脸特征库
=======================================
功能：读取 face_db/ 下每个人的照片，提取人脸特征编码，保存为 encodings.pkl
用法：python scripts/build_face_db.py

目录结构要求：
  face_db/
  ├── 张三/
  │   ├── 01.jpg
  │   ├── 02.jpg
  │   └── 03.jpg
  ├── 李四/
  │   ├── 01.jpg
  │   └── 02.jpg
  └── ...
"""

import face_recognition
import os
import pickle
import sys


def build_database(face_db_dir="face_db"):
    """扫描人脸目录，提取特征编码并保存"""

    if not os.path.exists(face_db_dir):
        print(f"错误：找不到人脸库目录 {face_db_dir}/")
        print("请创建该目录并为每个人建立子文件夹，放入 3~5 张正面照片")
        sys.exit(1)

    face_db = {}  # { '姓名': [encoding1, encoding2, ...] }
    total_images = 0
    failed_images = 0

    print("=" * 50)
    print("开始构建人脸特征库")
    print("=" * 50)

    for person_name in sorted(os.listdir(face_db_dir)):
        person_dir = os.path.join(face_db_dir, person_name)

        # 跳过文件（只处理文件夹）
        if not os.path.isdir(person_dir):
            continue

        # 跳过隐藏文件夹
        if person_name.startswith("."):
            continue

        encodings = []
        img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        for img_file in sorted(os.listdir(person_dir)):
            ext = os.path.splitext(img_file)[1].lower()
            if ext not in img_extensions:
                continue

            img_path = os.path.join(person_dir, img_file)
            total_images += 1

            try:
                # 加载图片并提取人脸编码
                image = face_recognition.load_image_file(img_path)
                encs = face_recognition.face_encodings(image)

                if encs:
                    encodings.append(encs[0])
                else:
                    print(f"  ⚠ {img_path}: 未检测到人脸，已跳过")
                    failed_images += 1
            except Exception as e:
                print(f"  ✗ {img_path}: 处理失败 ({e})")
                failed_images += 1

        if encodings:
            face_db[person_name] = encodings
            print(f"  ✓ {person_name}: {len(encodings)} 张照片已入库")
        else:
            print(f"  ✗ {person_name}: 无有效人脸照片，未入库")

    # 保存特征库
    if not face_db:
        print("\n错误：没有任何有效人脸数据！请检查 face_db/ 目录")
        sys.exit(1)

    output_path = os.path.join(face_db_dir, "encodings.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(face_db, f)

    print("\n" + "=" * 50)
    print(f"人脸特征库构建完成！")
    print(f"  注册人数  : {len(face_db)} 人")
    print(f"  处理图片  : {total_images} 张")
    print(f"  失败/跳过 : {failed_images} 张")
    print(f"  保存路径  : {output_path}")
    print("=" * 50)

    # 打印已注册人员名单
    print("\n已注册人员名单：")
    for i, name in enumerate(sorted(face_db.keys()), 1):
        print(f"  {i}. {name} ({len(face_db[name])} 张特征)")


if __name__ == "__main__":
    build_database()
