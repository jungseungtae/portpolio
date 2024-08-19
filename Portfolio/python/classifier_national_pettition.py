from glob_function import MLUtils as gf
import os
import pandas as pd
import numpy as np

## 콘솔창 설정
pd.set_option('display.max_columns', None)  # 모든 열을 표시
pd.set_option('display.unicode.east_asian_width', True)  # 한글 출력 시 정렬
pd.set_option('display.width', 1000)        # 콘솔 출력의 너비를 1000으로 설정

### 1. 데이터 불러오기
# df_origin = gf.load_csv('./data/petition.csv')

### 2. 데이터 탐색
# df = df_origin.dropna(subset = ['content'])
# gf.eda(df)
# df.info()

### 3. 데이터 분리(속도 저하로 샘플 생성)
# df = df.sample(frac = 0.1, random_state = 42)
# df.to_pickle('petition_sample.pkl')

df = pd.read_pickle('petition_sample.pkl')
# gf.eda(df)

df['start'] = pd.to_datetime(df['start'])
df['end'] = pd.to_datetime(df['end'])


### 4. 데이터분석

## 답변대상 생성
df['answer'] = (df['votes'] > 200000) == 1
# print(df.head())

## 기간분석
df['duration'] = df['end'] - df['start']
# print(df.sort_values('duration', ascending = True))

# duration_valc = df['duration'].value_counts()
# print(duration_valc)

# answer_30 = df[(df['duration'] == '30 days') & (df['answer'] == 1)]
# print(answer_30)

# start_date = df['start'].value_counts()
# print(start_date)

## 분야분석
# category = df['category'].value_counts()
# print(category)

## 카테고리별 답변여부와 투표수
pivot_df = df[['category', 'answer', 'answered', 'votes']]
petitions_unique = pd.pivot_table(pivot_df, index = ['category'], aggfunc = 'sum')
petition_best = petitions_unique.sort_values(by = 'votes', ascending = False).reset_index()
# print(petition_best)

## 청원일을 기준으로 투표수가 많은 일자
start_df = df[['start', 'answer', 'answered', 'votes']]
start_unique = pd.pivot_table(start_df, index = ['start'], aggfunc = 'sum')
start_best = start_unique.sort_values(by = 'votes', ascending = False).reset_index()
# print(start_best)

## 청원 등록이 가장 많은 일자
start_count = start_df.groupby('start').size().reset_index(name = 'counts')
start_count = start_count.sort_values(by = 'counts', ascending = False) #.reset_index(drop=True)
# print(start_count)

merged_df = start_df.merge(start_count, on = 'start', how = 'left')
merged_df = merged_df.sort_values(by = 'votes', ascending = False)
# print(merged_df)