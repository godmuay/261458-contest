import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# 🏗️ โหลด ResNet50 เป็น Feature Extractor
def create_feature_extractor():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    
    # ✅ ปลดล็อคบางชั้นให้ Train ได้
    for layer in base_model.layers[-30:]:  # Unfreeze 30 ชั้นสุดท้าย
        layer.trainable = True
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu')
    ])
    
    return model

# 🔄 โมเดลเปรียบเทียบภาพ (ใช้ Cosine Similarity)
def create_comparison_model():
    feature_extractor = create_feature_extractor()
    
    input_1 = Input(shape=(128, 128, 3))
    input_2 = Input(shape=(128, 128, 3))

    feature_1 = feature_extractor(input_1)
    feature_2 = feature_extractor(input_2)
    
    # ✅ ใช้ Cosine Similarity แทน Absolute Difference
    cosine_similarity = layers.Dot(axes=1, normalize=True)([feature_1, feature_2])
    
    combined = layers.Dense(128, activation='relu')(cosine_similarity)
    combined = layers.BatchNormalization()(combined)
    combined = layers.Dropout(0.5)(combined)
    output = layers.Dense(3, activation='softmax')(combined)  # เปลี่ยนเป็น 3 Class (1, 2, 3)
    
    model = models.Model(inputs=[input_1, input_2], outputs=output)
    model.compile(optimizer=AdamW(learning_rate=0.0001, weight_decay=1e-4), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

# 📂 โหลดข้อมูล
def load_data(csv_file, image_dir):
    df = pd.read_csv(csv_file)
    X1, X2, y = [], [], []
    
    for _, row in df.iterrows():
        img1_path = os.path.join(image_dir, row['Image 1'])
        img2_path = os.path.join(image_dir, row['Image 2'])
        
        if not os.path.exists(img1_path) or not os.path.exists(img2_path):
            print(f"⚠️ Missing {img1_path} or {img2_path}")
            continue
        
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        if img1 is None or img2 is None:
            print(f"⚠️ Cannot read {img1_path} or {img2_path}")
            continue
        
        img1 = cv2.resize(img1, (128, 128)) / 255.0
        img2 = cv2.resize(img2, (128, 128)) / 255.0
        X1.append(img1)
        X2.append(img2)
        y.append(row['Winner'])  # ให้เป็น 1, 2, 3 ตรงๆ
    
    return np.array(X1), np.array(X2), np.array(y)

# 📁 ตั้งค่าพาธไฟล์
dataset_csv = "D:/contest/Dataset_for_development-20250304T203703Z-004/Dataset_for_development/data_from_questionaire.csv"
image_dir = "D:/contest/Dataset_for_development-20250304T203703Z-004/Dataset_for_development/Questionair_Images"

# 📊 แบ่งข้อมูล Train/Test
X1, X2, y = load_data(dataset_csv, image_dir)
X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(X1, X2, y, test_size=0.2, random_state=42)

# 🏷️ ปรับสมดุลข้อมูล
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
print("📊 Class Weights:", class_weight_dict)

# 🎨 Data Augmentation
data_gen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    brightness_range=[0.7, 1.3],
    zoom_range=0.3,
    horizontal_flip=True
)

def augment_data(X1, X2, y):
    X1_aug, X2_aug, y_aug = [], [], []
    for i in range(len(X1)):
        img1_aug = data_gen.random_transform(X1[i])
        img2_aug = data_gen.random_transform(X2[i])
        X1_aug.append(img1_aug)
        X2_aug.append(img2_aug)
        y_aug.append(y[i])
    return np.array(X1_aug), np.array(X2_aug), np.array(y_aug)

X1_train, X2_train, y_train = augment_data(X1_train, X2_train, y_train)

# 🏗️ สร้างและเทรนโมเดล
model = create_comparison_model()
model.summary()

# 🔄 Learning Rate Scheduling
lr_schedule = LearningRateScheduler(lambda epoch: 1e-3 * 0.95 ** epoch)  # ใช้ Decay แทน

# 📌 บันทึกเฉพาะโมเดลที่ดีที่สุด
checkpoint = ModelCheckpoint('compare_model_best1111.keras', monitor='val_accuracy', save_best_only=True, mode='max')

# 🚀 เริ่ม Train
history = model.fit(
    [X1_train, X2_train], y_train,
    epochs=30, batch_size=20,
    validation_data=([X1_test, X2_test], y_test),
    class_weight=class_weight_dict,
    callbacks=[checkpoint, lr_schedule]
)

# 🎯 ทำนายผลลัพธ์จาก Test Set
y_pred = model.predict([X1_test, X2_test])
y_pred_classes = np.argmax(y_pred, axis=1)  # ใช้ argmax หาค่าที่โมเดลให้ความมั่นใจที่สุด

# 📌 Confusion Matrix
cm = confusion_matrix(y_test, y_pred_classes)

# 🔥 แสดง Confusion Matrix ด้วย Heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Class 1', 'Class 2', 'Class 3'], yticklabels=['Class 1', 'Class 2', 'Class 3'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# 📊 แสดง Classification Report
print("📌 Classification Report:\n", classification_report(y_test, y_pred_classes))
