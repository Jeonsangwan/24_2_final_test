import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from data_preprocessing import load_and_preprocess_images, split_data


# 모델 설계 (기존 코드 참조)
def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')  # 이진 분류
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model


# F1 점수 시각화 함수
def plot_f1_score(history, X_val, y_val):
    # 훈련 과정 동안 F1 점수 계산
    f1_scores = []

    for epoch in range(len(history.history['accuracy'])):
        # 현재 에폭에서의 예측값
        y_pred = (model.predict(X_val) > 0.5)  # 예측값이 0.5보다 크면 1로 분류

        # 정밀도, 재현율, F1 점수 계산
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)

        # F1 점수 리스트에 저장
        f1_scores.append(f1)

    # F1 점수 시각화
    plt.figure(figsize=(8, 6))
    plt.plot(f1_scores, label="F1 Score", color='blue', lw=2)
    plt.title('F1 Score during Training')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend(loc="lower right")
    plt.show()


# 데이터 준비
csv_path = '/Users/sangwanjeon/Documents/GitHub/24_2_final_test/updated_helmets.csv'
image_folder_path = '/Users/sangwanjeon/Documents/GitHub/24_2_final_test/png'

# 데이터 로드 및 전처리
images, labels = load_and_preprocess_images(csv_path, image_folder_path)

# 훈련, 검증, 테스트 데이터 분리
X_train, X_val, X_test, y_train, y_val, y_test = split_data(images, labels)

# 모델 빌드
model = build_model(X_train.shape[1:])

# 모델 훈련 및 history 저장
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# F1 점수 시각화 함수 호출
plot_f1_score(history, X_val, y_val)
