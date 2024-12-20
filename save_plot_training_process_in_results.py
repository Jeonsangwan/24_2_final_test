import pickle
import matplotlib.pyplot as plt
import os

# history.pkl 파일에서 훈련 과정 데이터 로드
with open('history.pkl', 'rb') as file:
    history = pickle.load(file)

# results 폴더 경로 설정
results_folder = '/Users/sangwanjeon/Documents/GitHub/24_2_final_test/results'

# 폴더가 존재하지 않으면 생성
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# 정확도 시각화
plt.figure(figsize=(12, 4))

# 정확도 그래프
plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label='Train Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# 손실 그래프
plt.subplot(1, 2, 2)
plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# 레이아웃 조정
plt.tight_layout()

# 결과를 results 폴더에 저장
output_path = os.path.join(results_folder, 'training_process.png')
plt.savefig(output_path)  # 이미지 저장

# 그래프 출력
plt.show()
