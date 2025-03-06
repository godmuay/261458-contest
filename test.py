import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras import models
import cv2
import numpy as np

# 📌 ฟังก์ชันเปรียบเทียบภาพ
def absolute_difference(tensors):
    return tf.abs(tensors[0] - tensors[1])

# 🚀 โหลดโมเดล
model = models.load_model("compare_model_besttt.keras", custom_objects={"absolute_difference": absolute_difference})

# 📂 กำหนดพาธของไฟล์ CSV และโฟลเดอร์รูปที่อยู่ในโฟลเดอร์เดียวกับ test.py
script_dir = os.path.dirname(os.path.abspath(__file__))  # หาโฟลเดอร์ที่ test.py อยู่
csv_file = os.path.join(script_dir, "test.csv")  # `test.csv` อยู่ในโฟลเดอร์เดียวกัน
image_folder = os.path.join(script_dir, "Test Images")  # `Test_Images` ก็อยู่โฟลเดอร์เดียวกัน

# ✅ ตรวจสอบว่าไฟล์ CSV และโฟลเดอร์รูปมีอยู่จริง
if not os.path.exists(csv_file):
    print(f"❌ ไม่พบไฟล์ CSV: {csv_file}")
    exit()

if not os.path.exists(image_folder):
    print(f"❌ ไม่พบโฟลเดอร์รูป: {image_folder}")
    exit()

print(f"✅ พบไฟล์ CSV: {csv_file}")
print(f"✅ พบโฟลเดอร์รูป: {image_folder}")

# 🔄 ฟังก์ชันทำนายว่าภาพไหนดูน่ากินกว่า
def predict_best_image(model, img1_path, img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None or img2 is None:
        print(f"❌ Error: ไม่พบไฟล์ {img1_path} หรือ {img2_path}")
        return None

    img1 = cv2.resize(img1, (128, 128)) / 255.0
    img2 = cv2.resize(img2, (128, 128)) / 255.0

    img1 = np.expand_dims(img1, axis=0)
    img2 = np.expand_dims(img2, axis=0)

    prediction = model.predict([img1, img2])

    return 1 if prediction[0][0] < 0.5 else 2
  # คืนค่า 1 ถ้า img1 ดีกว่า, 2 ถ้า img2 ดีกว่า

# 📌 อ่านไฟล์ CSV
df = pd.read_csv(csv_file)

# 🔄 วนลูปทุกแถวใน `test.csv`
for index, row in df.iterrows():
    img1_filename = row["Image 1"]
    img2_filename = row["Image 2"]

    img1_path = os.path.join(image_folder, img1_filename)
    img2_path = os.path.join(image_folder, img2_filename)

    # ตรวจสอบว่าไฟล์ภาพมีอยู่จริง
    if not os.path.exists(img1_path) or not os.path.exists(img2_path):
        print(f"❌ ไม่พบไฟล์ภาพ: {img1_path} หรือ {img2_path}")
        df.at[index, "Winner"] = "Error"
        continue

    # ทำนายภาพที่ดีกว่า
    winner = predict_best_image(model, img1_path, img2_path)
    
    if winner is not None:
        df.at[index, "Winner"] = winner
        print(f"✅ {img1_filename} 🆚 {img2_filename} → Winner: {winner}")

# 💾 บันทึกผลลัพธ์ลงใน `test.csv`
df.to_csv(csv_file, index=False)
print("\n✅ อัปเดตผลลัพธ์ลงในไฟล์ test.csv เรียบร้อย!")


