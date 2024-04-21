import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 데이터 불러오기
file_path = r'C:\Users\jstco\Downloads\hg-mldl-master\hg-mldl-master\fish.csv'
fish = pd.read_csv(file_path)

# 데이터 추출
bream_smelt_date = fish[fish['Species'].isin(['Bream', 'Smelt'])]
features = bream_smelt_date[['Length', 'Weight']]
target = bream_smelt_date['Species'].values

def extract_species_data(data, species):
    species_data = data[data['Species'] == species]
    length = species_data['Length'].values
    weight = species_data['Weight'].values
    return length, weight

# 각 변수에 할당
bream_length, bream_weight = extract_species_data(bream_smelt_date, 'Bream')
smelt_length, smelt_weight = extract_species_data(bream_smelt_date, 'Smelt')

# print(bream_length, bream_weight)
# print(smelt_length, smelt_weight)

import matplotlib.pyplot as plt

# plt.scatter(bream_length, bream_weight)
# plt.scatter(smelt_length, smelt_weight)
# plt.xlabel('length')
# plt.ylabel('weight')
# plt.show()

# print(features)

# bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
# bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]
#
# smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
# smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]
#
# length = bream_length + smelt_length
# weight = bream_weight + smelt_weight
#
# fish_data = [[l, w] for l, w in zip(length, weight)]
# print(fish_data)

# 데이터 타입 변환
bream_length = np.array(bream_length)
bream_weight = np.array(bream_weight)
smelt_length = np.array(smelt_length)
smelt_weight = np.array(smelt_weight)

length = np.concatenate([bream_length, smelt_length])
weight = np.concatenate([bream_weight, smelt_weight])

fish_data = np.column_stack([length, weight])
# print(fish_data)

fish_target = [1] * 35 + [0] * 14
# print(fish_target)

# K-근접 이웃 모델 적용
from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier()
kn.fit(fish_data, fish_target)
# print(kn.score(fish_data, fish_target))
# print(kn.predict([[30, 600]]))
# print(kn._fit_X)

kn49 = KNeighborsClassifier(n_neighbors=49)
kn49.fit(fish_data, fish_target)
print(kn49.score(fish_data, fish_target))