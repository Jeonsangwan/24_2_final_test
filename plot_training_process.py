import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from data_preprocessing import load_and_preprocess_images, split_data

# 모델 설계
def build_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # 이진 분류
    ])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

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

# 성능 평가
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")

# 모델 저장
model.save('helmet_detection_model.keras')

# history 객체 저장
import pickle
with open('history.pkl', 'wb') as file:
    pickle.dump(history.history, file)
