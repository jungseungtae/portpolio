import os
import pandas as pd
from sklearn.model_selection import train_test_split

# print(os.getcwd())
wine = pd.read_csv(os.getcwd() + '/data/wine.csv')

## 1. 탐색적 데이터분석
# wine.info()
# print(wine.head())
# print(wine.describe())

## 2. 데이터 전처리

# 분리
data = wine.drop('class', axis = 1)
target = wine['class']

train_input, test_input, train_target, test_target = train_test_split(data, target, test_size = 0.2, random_state=42)

# 표준화
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(train_input)

train_scaled = scaler.transform(train_input)
test_scaled = scaler.transform(test_input)
# print(train_scaler)

# col = data.columns
# print(col)
# data_scale = pd.DataFrame(train_scaled, columns = col)
# print(data_scale.head())
# print(data_scale.describe())

## 3. 결정트리 적용
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt.fit(train_scaled, train_target)

train_score = dt.score(train_scaled, train_target)
test_score = dt.score(test_scaled, test_target)

# print(f'train score : {round(train_score, 2)}')
# print(f'test score : {round(test_score, 2)}')

## 4. 트리 그래프 살펴보기
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize = (10, 7))
plot_tree(dt, max_depth = 1, filled = True, feature_names = ['alcohol', 'suger', 'pH'])
plt.show()