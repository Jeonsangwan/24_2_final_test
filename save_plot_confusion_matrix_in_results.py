import tensorflow as tf
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
from data_preprocessing import load_and_preprocess_images, split_data  # 데이터 전처리 함수 임포트

# 모델 로드, data_preprocessing에서 실행 후 파일명을 옮겨 적으면 됩니다.
model = tf.keras.models.load_model('helmet_detection_model.keras')

# 데이터 로드 및 전처리
csv_path = '/Users/sangwanjeon/Documents/GitHub/24_2_final_test/updated_helmets.csv'
image_folder_path = '/Users/sangwanjeon/Documents/GitHub/24_2_final_test/png'

# 데이터 준비
images, labels = load_and_preprocess_images(csv_path, image_folder_path)
X_train, X_val, X_test, y_train, y_val, y_test = split_data(images, labels)

# 예측 수행
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)  # 이진 분류이므로 임계값 0.5로 예측

# 혼동 행렬 계산
cm = confusion_matrix(y_test, y_pred)

# 결과 저장할 폴더 경로
results_folder = '/Users/sangwanjeon/Documents/GitHub/24_2_final_test/results'

# 폴더가 존재하지 않으면 생성
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# 혼동 행렬 시각화
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No Helmet", "Helmet"], yticklabels=["No Helmet", "Helmet"])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')

# 결과를 results 폴더에 저장
output_path = os.path.join(results_folder, 'confusion_matrix.png')
plt.savefig(output_path)  # 이미지 저장

# 그래프 출력
plt.show()
