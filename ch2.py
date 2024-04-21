import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file_path = r'C:\Users\jstco\Downloads\hg-mldl-master\hg-mldl-master\fish.csv'
fish = pd.read_csv(file_path)


## 물고기 종 분류하기

# 1. 데이터 대입하기
def extract_species_data(data, species):
    species_data = data[data['Species'] == species]
    length = species_data['Length'].values
    weight = species_data['Weight'].values
    return length, weight


bream_smelt_date = fish[fish['Species'].isin(['Bream', 'Smelt'])]
features = bream_smelt_date[['Length', 'Weight']]
target = bream_smelt_date['Species'].values

bream_length, bream_weight = extract_species_data(bream_smelt_date, 'Bream')
smelt_length, smelt_weight = extract_species_data(bream_smelt_date, 'Smelt')

fish_length = features['Length'].values
# print(fish_length)
fish_weight = features['Weight'].values

# fish_data = [[l, w] for l, w in zip(fish_length, fish_weight)]
# print(fish_data)
# fish_target = [1] * 35 + [0] * 14

# 2. 분류하기
from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier()

# 순서대로 하면 분류할 수 없음.
# print(fish_data[4])
# print(fish_data[:5])

# train_input = fish_data[:35]
# train_target = fish_target[:35]
# test_input = fish_data[35:]
# test_target = fish_target[35:]

# kn = kn.fit(train_input, train_target)
# print(kn.score(test_input, test_target))

# 데이터 섞기
# input_arr = np.array(fish_data)
# target_arr = np.array(fish_target)
# print(input_arr.shape)

# np.random.seed(42)
# index = np.arange(49)
# np.random.shuffle(index)
# print(index)

# train_input = input_arr[index[:35]]
# train_target = target_arr[index[:35]]
# test_input = input_arr[index[35:]]
# test_target = target_arr[index[35:]]

# 그래프로 확인하기
import matplotlib.pyplot as plt

# print(train_input[3, :1])
# plt.scatter(train_input[:, 0], train_input[:, 1], color = 'blue')
# plt.scatter(test_input[:, 0], test_input[:, 1], color ='red')
# plt.xlabel('length')
# plt.ylabel('weight')
# plt.show()

# print(train_input)
# print(test_input)

# print(train_target)
# print(test_target)

# kn = kn.fit(train_input, train_target)
# print(kn.score(test_input, test_target))

## 데이터 스케일 조정하기

# 다른 방법으로 데이터 만들기
fish_data = np.column_stack((fish_length, fish_weight))
fish_target = np.concatenate((np.ones(35), np.zeros(14)))
# print(fish_target)

# 데이터 나누기
from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, random_state=42, stratify=fish_target)
# print(train_input.shape, test_input.shape)
# print(type(train_input), type(test_input))

# 훈련하기
# kn.fit(train_input, train_target)
# kn.score(test_input, test_target)
# print(kn.predict([[25, 150]]))

# 오류값 표시
# plt.scatter(train_input[:, 0], train_input[:, 1])
# plt.scatter(25, 150, marker = '^', color = 'red')
# plt.xlabel('length')
# plt.ylabel('weight')
# plt.show()

# 확인하기
# distances, indexes = kn.kneighbors([[25, 150]])
# plt.scatter(train_input[indexes, 0], train_input[indexes, 1], marker='D')
# plt.show()
# print(train_input[indexes], train_target[indexes], distances)

# x축의 값 증가시켜 x, y축 비율 조정하기
# plt.xlim(0, 1000)
# plt.show()

# 평균과 편차로 계산하기
mean = np.mean(train_input, axis = 0)
std = np.std(train_input, axis = 0)
train_scaled = (train_input - mean) / std
# print(train_scaled)
# plt.scatter(25, 150, marker = '^', color = 'red')

new = ([25, 150] - mean) / std
plt.scatter(train_scaled[:, 0], train_scaled[:, 1])
plt.scatter(new[0], new[1], marker = '^')
# plt.show()

# 재훈련
kn.fit(train_scaled, train_target)
test_scaled = (test_input -mean) / std
kn.score(test_scaled, test_target)
# print(kn.predict([new]))

# 그래프로 확인
distances, indexes = kn.kneighbors([new])
plt.scatter(train_scaled[indexes, 0], train_scaled[indexes, 1], marker='D')
# plt.show()

