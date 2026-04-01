import argparse
import cv2
import os
from ultralytics import YOLO

def detect_image(image_path, save_path="output.jpg"):
    model = YOLO("yolo11n.pt")
    model.to('cpu')
    img = cv2.imread(image_path)
    results = model(img)
    annotated = results[0].plot()
    cv2.imwrite(save_path, annotated)
    print(f"✅ 检测完成，保存到：{save_path}")

def detect_dir(dir_path):
    out_dir = "detect_output"
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(dir_path):
        path = os.path.join(dir_path, f)
        detect_image(path, os.path.join(out_dir, f))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="单张图片路径")
    parser.add_argument("--dir", help="图片文件夹路径")
    args = parser.parse_args()
    if args.image:
        detect_image(args.image)
    elif args.dir:
        detect_dir(args.dir)
    else:
        print("⚠️ 请输入 --image 或 --dir 参数")