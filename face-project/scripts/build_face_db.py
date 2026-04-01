import os
import cv2
import pickle
from tqdm import tqdm

FACE_DB_PATH = "face_db"
ENCODING_PATH = os.path.join(FACE_DB_PATH, "encodings.pkl")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

known_features = []
known_names = []

for person_name in tqdm(os.listdir(FACE_DB_PATH), desc="构建人脸库"):
    person_dir = os.path.join(FACE_DB_PATH, person_name)
    if not os.path.isdir(person_dir):
        continue
    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)
        try:
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                face_roi = gray[y:y + h, x:x + w]
                resized_face = cv2.resize(face_roi, (32, 32))
                feature = resized_face.flatten()

                known_features.append(feature)
                known_names.append(person_name)
        except Exception as e:
            print(f"跳过 {img_path}: {e}")

data = {"features": known_features, "names": known_names}
with open(ENCODING_PATH, "wb") as f:
    pickle.dump(data, f)

print(f"✅ 人脸库构建完成！共 {len(set(known_names))} 人")