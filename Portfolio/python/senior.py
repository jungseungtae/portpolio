import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

file_path = r'C:\Users\jstco\OneDrive\바탕 화면\데이터베이스 데이터'
df = pd.read_csv(file_path + '/TB_JOBS_FOR_SENIOR.csv')

pd.set_option('display.max_columns', None)
plt.rcParams['font.family'] = 'Malgun Gothic'  # 'NanumGothic' 또는 'Malgun Gothic'
plt.rcParams['font.size'] = 12

# df.info()

# print(df[['CITY', 'TOTAL', 'TARGET_JOB']])

df = df[['CITY', 'TOTAL', 'TARGET_JOB']]

# corr_matrix = data[['TOTAL', 'TARGET_JOB']].corr()
# print(corr_matrix)

# TOTAL과 TARGET_JOB 컬럼 표준화
# df[['TOTAL', 'TARGET_JOB']] = scaler.fit_transform(df[['TOTAL', 'TARGET_JOB']])

# 표준화된 데이터 출력
# print(df[['CITY', 'TOTAL', 'TARGET_JOB']])

# 데이터 표준화
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[['TOTAL', 'TARGET_JOB']])
df[['TOTAL', 'TARGET_JOB']] = scaled_data

# 독립변수(X)와 종속변수(y) 분리
X = df[['TOTAL']]
y = df['TARGET_JOB']

# 선형 회귀 모델 생성 및 학습
model = LinearRegression()
model.fit(X, y)

# 예측 값 계산
df['PREDICTED_JOB'] = model.predict(X)

# 표준화된 값 역변환 (원래의 열들을 함께 변환)
df[['TOTAL', 'TARGET_JOB']] = scaler.inverse_transform(df[['TOTAL', 'TARGET_JOB']])
df['PREDICTED_JOB'] = df['PREDICTED_JOB'] * scaler.scale_[1] + scaler.mean_[1]

# 실제 값과 예측 값의 차이 계산
df['JOB_DIFFERENCE'] = df['TARGET_JOB'] - df['PREDICTED_JOB']

# 일자리 증감 필요성 계산
df['ADJUSTMENT'] = np.where(df['JOB_DIFFERENCE'] > 0, 'Decrease', 'Increase')
df['ADJUSTMENT_AMOUNT'] = df['JOB_DIFFERENCE'].abs()

print(df[['CITY', 'TARGET_JOB', 'PREDICTED_JOB', 'JOB_DIFFERENCE', 'ADJUSTMENT', 'ADJUSTMENT_AMOUNT']])


# 그래프 그리기
# plt.figure(figsize=(14, 8))
#
# # 현재 일자리 수와 예측된 일자리 수
# bar_width = 0.35
# index = np.arange(len(df['CITY']))
#
# bar1 = plt.bar(index, df['TARGET_JOB'], bar_width, label='Current Job')
# bar2 = plt.bar(index + bar_width, df['PREDICTED_JOB'], bar_width, label='Predicted Job')
#
# plt.xlabel('City')
# plt.ylabel('Number of Jobs')
# plt.title('Current and Predicted Job Numbers by City')
# plt.xticks(index + bar_width / 2, df['CITY'], rotation=45)
# plt.legend()
#
# # 일자리 증감 필요성 표시
# for i in range(len(df)):
#     plt.text(i, df['TARGET_JOB'][i] + 1000, f'{df["ADJUSTMENT"][i]}: {int(df["ADJUSTMENT_AMOUNT"][i])}',
#              ha='center', va='bottom', fontsize=10, color='black')
#
# plt.tight_layout()
# plt.show()
#