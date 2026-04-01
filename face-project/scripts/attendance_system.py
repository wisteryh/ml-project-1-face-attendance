import os
import cv2
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

# 加载人脸库
with open("face_db/encodings.pkl", "rb") as f:
    data = pickle.load(f)
known_features = np.array(data["features"])
known_names = data["names"]

# 检测器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
attended = set()
log_data = []

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (32, 32))
        feat = face_img.flatten().astype(np.float32)

        name = "Unknown"
        if len(known_features) > 0:
            distances = np.linalg.norm(known_features - feat, axis=1)
            idx = np.argmin(distances)
            if distances[idx] < 20000:
                name = known_names[idx]
                if name not in attended:
                    attended.add(name)
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_data.append([name, now])

        # 绿色框 + 正常名字
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 左上角正常显示（英文，无乱码）
    cv2.putText(frame, f"Attended: {len(attended)}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Face Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 保存记录
if log_data:
    os.makedirs("attendance_log", exist_ok=True)
    df = pd.DataFrame(log_data, columns=["Name", "Time"])
    df.to_csv(f"attendance_log/record_{datetime.now().strftime('%Y%m%d')}.csv", index=False, encoding="utf-8-sig")

cap.release()
cv2.destroyAllWindows()