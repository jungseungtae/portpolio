import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)  # 모든 열을 표시
pd.set_option('display.unicode.east_asian_width', True)  # 한글 출력 시 정렬
pd.set_option('display.width', 1000)        # 콘솔 출력의 너비를 1000으로 설정

import koreanize_matplotlib
from matplotlib_inline.backend_inline import set_matplotlib_formats
set_matplotlib_formats('retina')

################################################################

# 강수량 복구비용 예측하기

################################################################

# 1. 데이터 로드
file_path = r"C:\Users\jstco\OneDrive\바탕 화면\포트폴리오_강수량\전처리 데이터\csv_data"

# 강수량 데이터 로드
rain_fall_df = pd.read_csv(file_path + '/result.csv')
# print(rain_fall_df.describe())

# 홍수 위험도 데이터 로드
floor_risk_df = pd.read_csv(file_path + '/3. 홍수위험도_scaler.csv')
# print(floor_risk_df.info())

# 세종시는 측정값이 없으므로 제거하고, 불필요한 인덱스 컬럼 제거
floor_risk_df = floor_risk_df[floor_risk_df['city_code'] != '3600000000']
floor_risk_df.rename(columns={'city_code': 'CITY_CODE'}, inplace=True)
floor_risk_df.drop(columns=['Unnamed: 0'], axis=1, inplace=True)
# floor_risk_df.drop(columns=['household_farming'], axis=1, inplace=True)

# 2. 연도별 강수량 데이터에 리스크 데이터 추가하기
years = range(2012, 2022)
merged_df = pd.DataFrame()

# 연도별로 데이터 병합
for year in years:
    # 각 연도의 강수량 데이터 필터링
    result_year = rain_fall_df[rain_fall_df['YEAR'] == year]
    # 홍수 위험도 데이터와 병합
    merged_year = pd.merge(result_year, floor_risk_df, on='CITY_CODE', how='inner')
    # 병합된 데이터를 전체 데이터프레임에 추가
    merged_df = pd.concat([merged_df, merged_year], ignore_index=True)

city_names_test = merged_df[merged_df['YEAR'] == 2021]['CITY_NAME']

# 3. 데이터 전처리 및 데이터 셋 나누기
scaler = StandardScaler()

# 객체형 데이터를 제외하고, CITY_CODE 컬럼 제거
merged_df = merged_df.select_dtypes(exclude=['object']).drop(columns=['CITY_CODE'])

# 필요 컬럼만 표준화 진행
columns_to_scale = ['RAINY_DAYS', 'PRECIPITATION_DAY', 'TOTAL_PRECIPITATION', 'AMOUNT_DAMAGE', 'AMOUNT_RECOVERY']
merged_df[columns_to_scale] = scaler.fit_transform(merged_df[columns_to_scale])

# 2021년 데이터를 테스트 데이터로 설정하고 나머지는 학습 데이터로 설정
train_data = merged_df[merged_df['YEAR'] != 2021]
test_data = merged_df[merged_df['YEAR'] == 2021]
# train_data.info()

# 학습과 테스트 데이터로 분리
X_train = train_data.drop(columns=['YEAR', 'AMOUNT_RECOVERY'])
y_train = train_data['AMOUNT_RECOVERY']
X_test = test_data.drop(columns=['YEAR', 'AMOUNT_RECOVERY'])
y_test = test_data['AMOUNT_RECOVERY']
# print(X_train)

# 모델 리스트
# models = {
#     "Linear Regression": LinearRegression(),
#     "Ridge Regression": Ridge(alpha=1.0),
#     "Lasso Regression": Lasso(alpha=0.1),
# }
#
# # 성능 비교 결과 저장
# results = {}
#
# # 모델 학습 및 평가
# for name, model in models.items():
#     # 모델 학습
#     model.fit(X_train, y_train)
#     # 예측
#     y_pred = model.predict(X_test)
#
#     # 평가 지표 계산
#     mse = mean_squared_error(y_test, y_pred)
#     mae = mean_absolute_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)
#
#     # 결과 저장
#     results[name] = {"MSE": mse, "MAE": mae, "R2": r2}
#
#     # 결과 출력
#     print(f"{name} - "
#           f"Mean Squared Error: {mse:.2f}, "
#           f"Mean Absolute Error: {mae:.2f}, "
#           f"R-squared: {r2:.2f}")

# Mean Squared Error (MSE): 예측값과 실제값의 오차 제곱의 평균입니다.
# Mean Absolute Error (MAE): 예측값과 실제값의 오차의 절대값 평균으로, MSE보다 해석이 쉽고, 이상치의 영향을 덜 받습니다.
# R-squared (R²): 모델이 데이터를 얼마나 잘 설명하는지를 나타내는 지표로, 1에 가까울수록 모델이 잘 설명하고 있음을 의미합니다.

# Linear Regression - Mean Squared Error: 0.01, Mean Absolute Error: 0.07, R-squared: 0.92
# Ridge Regression - Mean Squared Error: 0.01, Mean Absolute Error: 0.07, R-squared: 0.89
# Lasso Regression - Mean Squared Error: 0.02, Mean Absolute Error: 0.04, R-squared: 0.80

# 4. 회귀 모델 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 5. 계수 추출 및 시각화
coefficients = model.coef_
features = X_train.columns

# 계수와 변수명을 데이터프레임으로 생성
coef_df = pd.DataFrame({'Feature': features, 'Coefficient': coefficients})

# 계수 크기 순으로 정렬
coef_df = coef_df.sort_values(by='Coefficient', ascending=False)

# 계수 시각화
plt.figure(figsize=(15, 6))
sns.barplot(x='Coefficient', y='Feature', data=coef_df)
plt.title('Linear Regression Coefficients')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.grid(True)
plt.show()

# 6. 결과 출력
print("각 독립변수가 복구비용에 미치는 영향:")
print(coef_df)


# 7. 예측 수행
y_pred = model.predict(X_test)

# 8. 평가 지표 계산
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 평가 결과 출력
# print(f"예측 값: {y_pred}")
# print(f"실제 값: {y_test.values}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R-squared: {r2:.2f}")

# 역표준화: scaler로 학습한 모든 컬럼을 사용하여 복구
test_data_scaled = test_data.copy()
test_data_scaled['AMOUNT_RECOVERY'] = y_pred  # 예측 값으로 교체
test_data_inverse = scaler.inverse_transform(test_data_scaled[columns_to_scale])

# 복구비용만 추출
y_test_actual = scaler.inverse_transform(test_data[columns_to_scale])[:, columns_to_scale.index('AMOUNT_RECOVERY')]
y_pred_actual = test_data_inverse[:, columns_to_scale.index('AMOUNT_RECOVERY')]


# 9. 결과 시각화
plt.figure(figsize=(14, 10))
plt.plot(city_names_test, y_test_actual, label='실제 복구 비용')
plt.plot(city_names_test, y_pred_actual, linestyle='--', label='예측된 복구 비용')
plt.title('2021년 실제 vs 예측된 복구 비용')
plt.xlabel('시도')  # x축 레이블을 시도로 설정
plt.ylabel('복구 비용 (천 원)')
plt.xticks(rotation=45, ha='right')  # 시도 이름이 잘 보이도록 회전
plt.legend()
plt.show()