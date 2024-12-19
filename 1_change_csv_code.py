import pandas as pd
import os

# 기존 CSV 파일 경로
csv_path = '/Users/sangwanjeon/Documents/GitHub/24_2_final_test/helmets.csv'

# 현재 CSV 파일 읽기
df = pd.read_csv(csv_path)

# 'helmet_on' 폴더와 'helmet_off' 폴더의 경로
helmet_on_folder = "/Users/sangwanjeon/Documents/GitHub/24_2_final_test/png/helmet_on"
helmet_off_folder = "/Users/sangwanjeon/Documents/GitHub/24_2_final_test/png/helmet_off"

# helmet_on 폴더의 이미지 파일들 (0.png부터 45.png까지)
helmet_on_images = [f'img/{i}.png' for i in range(46)]  # helmet_on -> 0.png부터 45.png까지

# helmet_off 폴더의 이미지 파일들 (0.png부터 11.png까지)
helmet_off_images = [f'img/{i}.png' for i in range(12)]  # helmet_off -> 0.png부터 11.png까지

# 새로운 CSV 내용 생성
new_data = []
for img in helmet_on_images:
    new_data.append([int(img.split('/')[1].split('.')[0]), img, 1])  # helmet_on -> 라벨 1
for img in helmet_off_images:
    new_data.append([int(img.split('/')[1].split('.')[0]), img, 0])  # helmet_off -> 라벨 0

# 새로운 DataFrame 생성
new_df = pd.DataFrame(new_data, columns=["image_id", "image_name", "label"])

# 저장할 경로 설정
new_csv_path = '/Users/sangwanjeon/Documents/GitHub/24_2_final_test/updated_helmets.csv'

# 새로운 CSV 파일 저장
new_df.to_csv(new_csv_path, index=False)

print(f"Updated CSV saved at: {new_csv_path}")
