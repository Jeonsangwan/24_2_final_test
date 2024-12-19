import pandas as pd
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import numpy as np


def load_and_preprocess_images(csv_path, image_folder_path, target_size=(128, 128)):
    # CSV 파일 읽기
    df = pd.read_csv(csv_path)

    images = []
    labels = []

    # 이미지 경로와 라벨을 읽어서 데이터에 추가
    for index, row in df.iterrows():
        # 이미 수정된 경로에 맞게 이미지 경로 생성
        img_path = os.path.join(image_folder_path, row['image_name'])  # helmet_on/ 또는 helmet_off/ 경로로 변경
        label = row['label']

        # 이미지 로드 및 전처리
        img = load_img(img_path, target_size=target_size)
        img_array = img_to_array(img) / 255.0  # 정규화

        images.append(img_array)
        labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    return images, labels


def split_data(images, labels, test_size=0.2, val_size=0.1):
    # 훈련, 검증, 테스트 데이터 분할 (80% 훈련, 10% 검증, 10% 테스트)
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=test_size, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    # CSV 파일 경로와 이미지 폴더 경로
    csv_path = '/Users/sangwanjeon/Documents/GitHub/24_2_final_test/updated_helmets.csv'
    image_folder_path = '/Users/sangwanjeon/Documents/GitHub/24_2_final_test/png'  # 이미지 경로는 png 폴더로

    # 이미지와 라벨 로드
    images, labels = load_and_preprocess_images(csv_path, image_folder_path)

    # 훈련, 검증, 테스트 데이터 분리
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(images, labels)

    # 데이터셋 크기 확인
    print(
        f"Training data shape: {X_train.shape}, Validation data shape: {X_val.shape}, Test data shape: {X_test.shape}")
