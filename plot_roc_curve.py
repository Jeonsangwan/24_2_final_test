import tensorflow as tf
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from data_preprocessing import load_and_preprocess_images, split_data  # 데이터 전처리 함수 임포트

# 모델 로드
model = tf.keras.models.load_model('helmet_detection_model.keras')

# 데이터 로드 및 전처리
csv_path = '/Users/sangwanjeon/Documents/GitHub/24_2_final_test/updated_helmets.csv'
image_folder_path = '/Users/sangwanjeon/Documents/GitHub/24_2_final_test/png'

# 데이터 준비
images, labels = load_and_preprocess_images(csv_path, image_folder_path)
X_train, X_val, X_test, y_train, y_val, y_test = split_data(images, labels)

# 예측 수행
y_pred = model.predict(X_test)  # 테스트 데이터 예측
y_pred = y_pred.flatten()  # 이진 분류를 위해 예측을 1D로 펼침

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# ROC Curve 시각화
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')  # 랜덤 모델
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
