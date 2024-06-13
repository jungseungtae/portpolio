### 회귀 분석
import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

# 데이터
perch_length = np.array(
    [8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0,
     21.0, 21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5,
     22.5, 22.7, 23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5,
     27.3, 27.5, 27.5, 27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0,
     36.5, 36.0, 37.0, 37.0, 39.0, 39.0, 39.0, 40.0, 40.0, 40.0,
     40.0, 42.0, 43.0, 43.0, 43.5, 44.0]
)
perch_weight = np.array(
    [5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0,
     110.0, 115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0,
     130.0, 150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0,
     197.0, 218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0,
     514.0, 556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0,
     820.0, 850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0,
     1000.0, 1000.0]
)

# 훈련 세트와 테스트 세트로 분리
train_input, test_input, train_target, test_target = train_test_split(
    perch_length, perch_weight, random_state=42
)

# print(train_input)
# 2차원 배열로 변경
train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)
# print(train_input)

# 데이터 산점도
# plt.scatter(perch_length, perch_weight)
# plt.xlabel("length")
# plt.ylabel("weight")
# plt.title("data scatter")
# plt.show()

## 1. 선형 회귀

# 선형 회귀 모델 훈련
lr = LinearRegression()
lr.fit(train_input, train_target)

# 길이가 50인 물고기의 무게 예측값
# print(lr.predict([[50]]))

# 기울기와 절편값
# print(lr.coef_, lr.intercept_)

# 회귀 직선 찾기
# plt.scatter(train_input, train_target)
# plt.plot([15, 50], [15 * lr.coef_ + lr.intercept_, 50 * lr.coef_ + lr.intercept_])
# plt.scatter(50, 1241.8, marker='^')
# plt.title("linear regression line")
# plt.show()

# 회귀직선의 R2 스코어
print('회귀직선 train R2 스코어', lr.score(train_input, train_target))
print('회귀직선 test R2 스코어', lr.score(test_input, test_target))

## 2. 다항 회귀
# 제곱한 수치를 추가
train_poly = np.column_stack((train_input ** 2, train_input))
test_poly = np.column_stack((test_input ** 2, test_input))

lr.fit(train_poly, train_target)
# print(lr.coef_, lr.intercept_)

# 회귀 곡선 찾기
# point = np.arange(15, 50)
# plt.scatter(train_input, train_target)
# plt.plot(point, 1.01 * point ** 2 - 21.6 * point + 116.05)
# plt.scatter([50], [1574], marker='^')
# plt.title("non-linear regression line")
# plt.show()

# 회귀곡선의 R2 스코어
print('회귀곡선 train R2 스코어', lr.score(train_poly, train_target))
print('회귀곡선 test R2 스코어', lr.score(test_poly, test_target))

## 3. 다중 회귀

# 다중 데이터 준비
df = pd.read_csv(os.getcwd() + '/data/perch_full.csv')
print(df.describe())
perch_data = df.to_numpy()
# print(perch_data)

# 훈련, 테스트 세트로 분리
train_input, test_input, train_target, test_target = train_test_split(
    perch_data, perch_weight, random_state=42
)

# 특성 만들기
poly = PolynomialFeatures()
poly.fit([[2, 3]])

poly.fit(train_input)
train_poly = poly.transform(train_input)
# print(poly.get_feature_names_out())

test_poly = poly.transform(test_input)

lr.fit(train_poly, train_target)
# print(lr.coef_, lr.intercept_)

print('다중 회귀 train R2 스코어', lr.score(train_poly, train_target))
print('다중 회귀 test R2 스코어', lr.score(test_poly, test_target))

## 규제

# 과대적합 만들기(특성 개수 증가 시키기)
poly = PolynomialFeatures(degree=5)
poly.fit(train_input)
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)
# print(train_poly.shape)
lr.fit(train_poly, train_target)
print('과대적합 train R2 스코어', lr.score(train_poly, train_target))
print('과대적합 test R2 스코어', lr.score(test_poly, test_target))

# 표준화
ss = StandardScaler()
ss.fit(train_poly)
train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)

# 릿지 회귀
# ridge = Ridge()
# ridge.fit(train_scaled, train_target)
# print('릿지 train 스코어', ridge.score(train_scaled, train_target))
# print('릿지 test 스코어', ridge.score(train_scaled, train_target))

# 릿지 하이퍼파라미터 찾기
train_score = []
test_score = []

alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
# for alpha in alpha_list:
#     ridge = Ridge(alpha=alpha)
#     ridge.fit(train_scaled, train_target)
#     train_score.append(ridge.score(train_scaled, train_target))
#     test_score.append(ridge.score(test_scaled, test_target))
#
# plt.plot(np.log10(alpha_list), train_score)
# plt.plot(np.log10(alpha_list), test_score)
# plt.title("Ridge Score")
# plt.show()


## 라쏘 회귀
lasso = Lasso()
lasso.fit(train_scaled, train_target)
print('라쏘 train 스코어', lasso.score(train_scaled, train_target))
print('라쏘 test 스코어', lasso.score(test_scaled, test_target))

train_score = []
test_score = []
alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]

# 라쏘 파라미터 찾기
for alpha in alpha_list:
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(train_scaled, train_target)
    train_score.append(lasso.score(train_scaled, train_target))
    test_score.append(lasso.score(test_scaled, test_target))

plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.title("Lasso Score")
plt.show()

lasso = Lasso(alpha=10)
lasso.fit(train_scaled, train_target)
print('라쏘 train 스코어', lasso.score(train_scaled, train_target))
print('라쏘 test 스코어', lasso.score(test_scaled, test_target))


# 제거된 특성
print('lasso value 0 : ', np.sum(lasso.coef_ == 0))
print('lasso value 0 < : ', np.sum(lasso.coef_ > 0))