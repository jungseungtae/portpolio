import glob_function as gf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib_inline.backend_inline import set_matplotlib_formats
from wordcloud import WordCloud
from konlpy.tag import Okt
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, GridSearchCV, cross_val_score
import koreanize_matplotlib
from sklearn.preprocessing import label_binarize
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import os
import re

# 한글 폰트 설정 및 그래프 스타일 설정
set_matplotlib_formats('retina')
# sns.set(style="whitegrid")
tqdm.pandas()

# print(os.getcwd())

### 1. 데이터 불러오기 ###
train = pd.read_csv('./data/train_data.csv')
test = pd.read_csv('./data/test_data.csv')
topic = pd.read_csv('./data/topic_dict.csv')

raw = pd.concat([train, test])
df = raw.merge(topic, how='left')
# print(df.info())
# print(df.head())


### 2. 데이터 탐색 ###
# gf.MLUtils.eda(train)
# gf.MLUtils.eda(test)
# gf.MLUtils.eda(topic)
# print(topic)

### 3. 데이터 전처리 ###
# 소문자 변환 및 특수문자 제거, 불용어 제거
def preprocess_text(text):
    text = text.lower()  # 소문자 변환
    text = re.sub(r'[^가-힣a-zA-Z0-9\s]', '', text)  # 특수문자 제거
    return text

# df['title'] = df['title'].apply(preprocess_text)
# print(df.head())

## 불용어 제거
okt = Okt()
def okt_clean(text):
    clean_text = []
    for word in okt.pos(text, stem=True):
        if word[1] not in ['Josa', 'Eomi', 'Punctuation']:
            clean_text.append(word[0])
    return ' '.join(clean_text)

# df['title'] = df['title'].progress_map(okt_clean)


def remove_stopwords(text):
    tokens = text.split(' ')
    stops = ['합니다', '하는', '할', '하고', '한다',
             '그리고', '입니다', '그', '등', '이런', '것 ', '및 ','제', '더',
             '하다', '되다', '없다']
    meaningful_words = [w for w in tokens if not w in stops]
    return ' '.join(meaningful_words)

# df['title'] = df['title'].map(remove_stopwords)

# df.to_pickle('df_preprocessing.pkl')
df = pd.read_pickle('data/df_preprocessing.pkl')
# print(df.head())

label_name = 'topic_idx'

### 데이터 분리 ###
train = df[df[label_name].notnull()]
test = df[df[label_name].isnull()]
X_train = train['title']
X_test = test['title']
y_train = train[label_name]

# print(X_train.head())


# ### 4. 토큰화, 벡터화 ###
tfidf_vect = TfidfVectorizer(tokenizer=None,
                             ngram_range=(1, 2), # 단일 단어, 2개의 단어 연속 단어
                             min_df=3,           # 최소 3개 문서에 등장
                             max_df=0.95)        # 전체 문서의 95% 이상에서 등장하면 제외(빈번한 단어)
tfidf_vect.fit(X_train)
train_feature_tfidf = tfidf_vect.transform(X_train)
test_feature_tfidf = tfidf_vect.transform(X_test)
# print(train_feature_tfidf)
'''
(문서 인덱스, 단어 인덱스, TF-IDF 값)
(0, 26734)	0.33292632511027165
(0, 25614)	0.3692059845314441
TF-IDF : 해당 문서에서 특정 단어의 가중치. 높을수록 중요한 단어로 판단
TF (Term Frequency): 문서에서 특정 단어가 얼마나 자주 등장하는지 나타냅니다.
IDF (Inverse Document Frequency): 특정 단어가 얼마나 희귀한지를 나타냅니다. 문서 내에서 자주 등장하는 단어일수록 IDF 값이 낮아집니다.
최종 TF-IDF 값은 TF와 IDF 값을 곱한 값입니다. 따라서 높은 TF-IDF 값은 해당 문서에서 해당 단어의 중요도.
'''


### 5. 모델적용 ###
# (1) 모델 정의
# models = {
#     'RandomForest' : RandomForestClassifier(random_state=42, n_jobs=-1),
#     'LGBM' : LGBMClassifier(random_state = 42, n_estimators = 100, n_jobs = -1),
#     'XGB' : XGBClassifier(random_state = 42, n_estimators = 100, n_jobs = -1)
# }

# (2) 정확도 계산
# accuracy_scores = {}
# for model_name, model in models.items():
#     print(f'Training {model_name}')
#     accuracy = cross_val_score(model,
#                                train_feature_tfidf,
#                                y_train,
#                                cv = 3,
#                                scoring='accuracy',
#                                n_jobs=-1,
#                                verbose=1).mean()
#     accuracy_scores[model_name] = accuracy
#     print(f'{model_name} Accuracy : {accuracy :.4f}')
#
# best_model_name = max(accuracy_scores, key = accuracy_scores.get)
# best_model_accuracy = accuracy_scores[best_model_name]

# print(f'\nBest Model : {best_model_name} with Accuracy : {best_model_accuracy:4f}')

'''
RandomForest Accuracy : 0.7665
LGBM Accuracy : 0.7806
XGB Accuracy : 0.7618

Best Model : LGBM with Accuracy : 0.780611
'''

# model = RandomForestClassifier(random_state=42, n_jobs=-1)
model = LGBMClassifier(random_state = 42, n_estimators = 100, n_jobs = -1)
# model = XGBClassifier(random_state = 42, n_estimators = 100, n_jobs = -1)

### 모델 학습 및 예측 ###
# model.fit(train_feature_tfidf, y_train)
# y_pred = cross_val_predict(model, train_feature_tfidf, y_train, cv=3, n_jobs=-1)
#
# cm = confusion_matrix(y_train, y_pred)

## confusion image
labels = ['IT', '경제', '사회', '생활', '세계', '스포츠', '정치']

# plt.figure(figsize=(10, 8))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.savefig('confusion_matrix.png')
# plt.show()

# 세부 모델 평가
# report = classification_report(y_train, y_pred, target_names=labels)
# print(report)

## 이미지 생성
font_path = 'C:/Windows/Fonts/malgun.ttf'

### 워드클라우드 생성 ###
# all_text = ' '.join(df['title'])
# wordcloud = WordCloud(font_path=font_path, background_color='white', width=800, height=400).generate(all_text)
#
# plt.figure(figsize=(15, 10))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
# plt.title('Word Cloud of Titles')
# plt.show()


# ### 토픽 이미지
# sns.countplot(data=df, x='topic')
# plt.xticks(rotation=45)
# plt.title('Topic Distribution')
# plt.show()


### 피처 횟수
# vocab = tfidf_vect.get_feature_names_out()
# dist = np.sum(train_feature_tfidf, axis=0)
# vocab_count = pd.DataFrame(dist, columns=vocab).T
# vocab_count.columns = ['count']
# vocab_count = vocab_count.sort_values(by='count', ascending=False).head(30)
#
# plt.figure(figsize=(15, 10))
# vocab_count['count'].plot(kind='bar')
# plt.title('Top 30 Features by Frequency')
# plt.xlabel('Feature Names')
# plt.xticks(rotation=45)
# plt.ylabel('Frequency')
# plt.show()


# ## 토픽별 상위 단어
tfidf_vect = TfidfVectorizer(tokenizer=None, ngram_range=(1, 2), min_df=3, max_df=0.95)
tfidf_vect.fit(df['title'])
tfidf_matrix = tfidf_vect.transform(df['title'])
feature_names = tfidf_vect.get_feature_names_out()

# # 각 토픽별로 TF-IDF 상위 단어 추출 및 시각화
# for topic_idx, topic_name in enumerate(df['topic'].unique()):
#     topic_data = df[df['topic'] == topic_name]
#
#     if topic_data.shape[0] == 0:
#         print(f"No data for topic '{topic_name}'")
#         continue
#
#     topic_tfidf_matrix = tfidf_vect.transform(topic_data['title'])
#
#     # 각 단어의 TF-IDF 합계 계산
#     topic_tfidf_sum = np.sum(topic_tfidf_matrix, axis = 0)
#     topic_tfidf_sum = np.squeeze(np.asarray(topic_tfidf_sum))
#
#     if topic_tfidf_sum.sum() == 0:
#         print(f"No meaningful words for topic '{topic_name}'")
#         continue
#
#     # 상위 10개 단어 추출
#     top_n = 10
#     top_n_indices = topic_tfidf_sum.argsort()[-top_n:][::-1]
#     top_n_words = [feature_names[i] for i in top_n_indices]
#     top_n_scores = [topic_tfidf_sum[i] for i in top_n_indices]
#
#     print(f"Top words for topic '{topic_name}':")
#     for word, score in zip(top_n_words, top_n_scores):
#         print(f"{word}: {score:.4f}")
#
#     # 워드클라우드 생성
#     word_freq = {word: score for word, score in zip(top_n_words, top_n_scores)}
#     wordcloud = WordCloud(
#         font_path = 'C:/Windows/Fonts/malgun.ttf',
#         background_color = 'white',
#         width = 800,
#         height = 400
#         ).generate_from_frequencies(word_freq)
#
#     plt.figure(figsize = (10, 5))
#     plt.imshow(wordcloud, interpolation = 'bilinear')
#     plt.axis('off')
#     plt.title(f'Word Cloud for Topic: {topic_name}')
#     plt.show()
#
#     # 막대그래프 생성
#     plt.figure(figsize=(10, 5))
#     plt.barh(top_n_words, top_n_scores, color='skyblue')
#     plt.xlabel('TF-IDF Score')
#     plt.title(f'Top {top_n} Words for Topic: {topic_name}')
#     plt.gca().invert_yaxis()  # y축을 반전시켜 가장 높은 점수가 위로 오게 함
#     plt.show()

# 경로 설정 (저장할 디렉토리)
save_path = r'C:\Users\jstco\OneDrive\바탕 화면\포트폴리오 뉴스제목 분석\포폴'  # 저장할 디렉토리 경로를 설정하세요.
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 각 토픽별로 TF-IDF 상위 단어 추출 및 시각화
for topic_idx, topic_name in enumerate(df['topic'].unique()):
    topic_data = df[df['topic'] == topic_name]

    if topic_data.shape[0] == 0:
        print(f"No data for topic '{topic_name}'")
        continue

    topic_tfidf_matrix = tfidf_vect.transform(topic_data['title'])

    # 각 단어의 TF-IDF 합계 계산
    topic_tfidf_sum = np.sum(topic_tfidf_matrix, axis=0)
    topic_tfidf_sum = np.squeeze(np.asarray(topic_tfidf_sum))

    if topic_tfidf_sum.sum() == 0:
        print(f"No meaningful words for topic '{topic_name}'")
        continue

    # 상위 10개 단어 추출
    top_n = 10
    top_n_indices = topic_tfidf_sum.argsort()[-top_n:][::-1]
    top_n_words = [feature_names[i] for i in top_n_indices]
    top_n_scores = [topic_tfidf_sum[i] for i in top_n_indices]

    print(f"Top words for topic '{topic_name}':")
    for word, score in zip(top_n_words, top_n_scores):
        print(f"{word}: {score:.4f}")

    # 워드클라우드 생성 및 저장
    word_freq = {word: score for word, score in zip(top_n_words, top_n_scores)}
    wordcloud = WordCloud(
        font_path='C:/Windows/Fonts/malgun.ttf',
        background_color='white',
        width=800,
        height=400
    ).generate_from_frequencies(word_freq)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for Topic: {topic_name}')

    # 워드클라우드 이미지 저장
    wordcloud_path = os.path.join(save_path, f'wordcloud_{topic_name}.png')
    plt.savefig(wordcloud_path, format='png')
    plt.close()  # 저장 후 닫기

    # 막대그래프 생성 및 저장
    plt.figure(figsize=(10, 5))
    plt.barh(top_n_words, top_n_scores, color='skyblue')
    plt.xlabel('TF-IDF Score')
    plt.title(f'Top {top_n} Words for Topic: {topic_name}')
    plt.gca().invert_yaxis()  # y축을 반전시켜 가장 높은 점수가 위로 오게 함

    # 막대그래프 이미지 저장
    bargraph_path = os.path.join(save_path, f'bargraph_{topic_name}.png')
    plt.savefig(bargraph_path, format='png')
    plt.close()  # 저장 후 닫기