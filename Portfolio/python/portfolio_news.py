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
from sklearn.model_selection import cross_val_predict, GridSearchCV
import koreanize_matplotlib
from sklearn.preprocessing import label_binarize
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


# 한글 폰트 설정 및 그래프 스타일 설정
set_matplotlib_formats('retina')
# sns.set(style="whitegrid")
tqdm.pandas()

### 1. 데이터 불러오기 ###
file_path = r'C:\Users\jstco\Downloads\pytextbook-main\pytextbook-main\data\klue'
train_data = file_path + '/train_data.csv'
test_data = file_path + '/test_data.csv'
topic_data = file_path + '/topic_dict.csv'

train = pd.read_csv(train_data)
test = pd.read_csv(test_data)
topic = pd.read_csv(topic_data)

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
## 정규화
df['title'] = df['title'].str.replace('[0-9]', '', regex=True)
df['title'] = df['title'].str.lower()

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
             '그리고', '입니다', '그', '등', '이런', '것 ', '및 ','제', '더']
    meaningful_words = [w for w in tokens if not w in stops]
    return ' '.join(meaningful_words)

# df['title'] = df['title'].map(remove_stopwords)

# df.to_pickle('df_preprocessing.pkl')
# df = pd.read_pickle('df_preprocessing.pkl')
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
tfidf_vect = TfidfVectorizer(tokenizer=None, ngram_range=(1, 2), min_df=3, max_df=0.95)
tfidf_vect.fit(X_train)
train_feature_tfidf = tfidf_vect.transform(X_train)
test_feature_tfidf = tfidf_vect.transform(X_test)
# print(train_feature_tfidf)


### 5. 모델적용 ###
# model = RandomForestClassifier(random_state=42, n_jobs=-1)
model = LGBMClassifier(random_state = 42, n_estimators = 100, n_jobs = -1)
# model = XGBClassifier(random_state = 42, n_estimators = 100, n_jobs = -1)

### 교차 검증을 통한 예측 ###
y_pred = cross_val_predict(model, train_feature_tfidf, y_train, cv=3, verbose=1, n_jobs=-1)

# y_pred.to_csv('y_pred_processed.csv', index=False)      # 랜덤포레스트

# y_pred_df = pd.read_pickle('y_pred_processed.pkl')
# y_pred_df = pd.DataFrame(y_pred_df, columns=['pred'])
# y_pred_series = y_pred_df['pred']

# valid_accuracy = (y_pred_series == y_train).mean()
# valid_accuracy = (y_pred == y_train).mean()
valid_accuracy = accuracy_score(y_train, y_pred)
print(f'Validation Accuracy: {valid_accuracy:.2f}')
print()

### 모델 학습 및 예측 ###
# model.fit(train_feature_tfidf, y_train)
# y_predict = model.predict(test_feature_tfidf)

# best_model.fit(train_feature_tfidf, y_train)


# sub = './result.csv'
# submit = pd.read_csv(sub)
# print(submit.head())
# submit['pred_idx'] = y_predict
# submit.to_csv('result.csv', index=False)

### 6. 모델평가
cm = confusion_matrix(y_train, y_pred)
print("Confusion Matrix:")
print(cm)

## confusion image
labels = ['IT', '경제', '사회', '생활', '세계', '스포츠', '정치']

# plt.figure(figsize=(10, 8))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.savefig('confusion_matrix.png')
# plt.show()

print(f"\nClassification Report: ")
# print(classification_report(y_train, y_pred_series))
print(classification_report(y_train, y_pred))


report = classification_report(y_train, y_pred, target_names=labels)
print(report)

# y_pred = cross_val_predict(model, train_feature_tfidf, y_train, cv=3, method='predict_proba')
# y_pred_classes = np.argmax(y_pred, axis=1)
#
# y_train_binarized = label_binarize(y_train, classes=[0, 1, 2, 3, 4, 5, 6])
# n_classes = y_train_binarized.shape[1]
#
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
#
# for i in range(n_classes):
#     fpr[i], tpr[i], _ = roc_curve(y_train_binarized[:, i], y_pred[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])
#
# # Macro-average ROC curve and ROC area
# all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# mean_tpr = np.zeros_like(all_fpr)
# for i in range(n_classes):
#     mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
#
# mean_tpr /= n_classes
# fpr["macro"] = all_fpr
# tpr["macro"] = mean_tpr
# roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
#
# # Plot ROC curves
# plt.figure(figsize=(10, 8))
# colors = ['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown']
# for i, color in zip(range(n_classes), colors):
#     plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC curve of class {i} (area = {roc_auc[i]:0.2f})')
#
# plt.plot(fpr["macro"], tpr["macro"], color='navy', linestyle=':', linewidth=4, label=f'Macro-average ROC curve (area = {roc_auc["macro"]:0.2f})')
#
# plt.plot([0, 1], [0, 1], 'k--', lw=2)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic - Multi-Class')
# plt.legend(loc="lower right")
# plt.show()


### 이미지 생성
# font_path = 'C:/Windows/Fonts/malgun.ttf'

# ### 워드클라우드 생성 ###
# all_text = ' '.join(df['title'])
# wordcloud = WordCloud(font_path=font_path, background_color='white', width=800, height=400).generate(all_text)

# plt.figure(figsize=(15, 10))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
# plt.title('Word Cloud of Titles')
# plt.show()


# ### 토픽 이미지
# sns.countplot(data=df, x='topic')
# plt.xticks(rotation=90)
# plt.title('Topic Distribution')
# plt.show()


# ### 피처 횟수
# vocab = tfidf_vect.get_feature_names_out()
# dist = np.sum(train_feature_tfidf, axis=0)
# vocab_count = pd.DataFrame(dist, columns=vocab).T
# vocab_count.columns = ['count']
# vocab_count = vocab_count.sort_values(by='count', ascending=False).head(50)
#
# plt.figure(figsize=(15, 10))
# vocab_count['count'].plot(kind='bar')
# plt.title('Top 50 Features by Frequency')
# plt.xlabel('Feature Names')
# plt.ylabel('Frequency')
# plt.show()


### 토픽별 상위 단어

# tfidf_vect = TfidfVectorizer(tokenizer=None, ngram_range=(1, 2), min_df=3, max_df=0.95)
# tfidf_vect.fit(df['title'])
# tfidf_matrix = tfidf_vect.transform(df['title'])
# feature_names = tfidf_vect.get_feature_names_out()
#
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
#     # word_freq = {word: score for word, score in zip(top_n_words, top_n_scores)}
#     # wordcloud = WordCloud(
#     #     font_path = 'C:/Windows/Fonts/malgun.ttf',
#     #     background_color = 'white',
#     #     width = 800,
#     #     height = 400
#     #     ).generate_from_frequencies(word_freq)
#     #
#     # plt.figure(figsize = (10, 5))
#     # plt.imshow(wordcloud, interpolation = 'bilinear')
#     # plt.axis('off')
#     # plt.title(f'Word Cloud for Topic: {topic_name}')
#     # plt.show()
#
#     # 막대그래프 생성
#     plt.figure(figsize=(10, 5))
#     plt.barh(top_n_words, top_n_scores, color='skyblue')
#     plt.xlabel('TF-IDF Score')
#     plt.title(f'Top {top_n} Words for Topic: {topic_name}')
#     plt.gca().invert_yaxis()  # y축을 반전시켜 가장 높은 점수가 위로 오게 함
#     plt.show()