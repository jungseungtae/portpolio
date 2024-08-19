import re
import pandas as pd
from konlpy.tag import Okt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, auc, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import numpy as np
from sklearn.preprocessing import label_binarize
from lightgbm import LGBMClassifier
from matplotlib_inline.backend_inline import set_matplotlib_formats
from tqdm import tqdm
import glob_function as gf
import matplotlib.font_manager as fm
from matplotlib import rc

# 한글 폰트 설정 및 그래프 스타일 설정
set_matplotlib_formats('retina')
font_path = 'C:/Windows/Fonts/malgun.ttf'
fontprop = fm.FontProperties(fname=font_path)
rc('font', family=fontprop.get_name())
# sns.set(style="whitegrid")
tqdm.pandas()

# 데이터 로딩
train_data = gf.MLUtils.load_csv('data/train_data.csv')
test_data = gf.MLUtils.load_csv('data/test_data.csv')
topic_data = gf.MLUtils.load_csv('data/topic_dict.csv')

# 데이터 병합
merge_data = train_data.merge(topic_data, on = 'topic_idx')

# merge_data.info()
# print(merge_data)

# 특수문자, 숫자 제거, 소문자 변환
merge_data['title'] = merge_data['title'].str.replace(r'[^가-힣\s]', '')
merge_data['title'] = merge_data['title'].str.lower()

# 불용어 처리
okt = Okt()

def okt_clean(text):
    clean_text = []
    for word in okt.pos(text, stem=True):
        if word[1] not in ['Josa', 'Eomi', 'Punctuation']:
            clean_text.append(word[0])
    return ' '.join(clean_text)

merge_data['title'] = merge_data['title'].progress_map(okt_clean)

def remove_stopwords(text):
    tokens = text.split(' ')
    stops = ['합니다', '하는', '할', '하고', '한다',
             '그리고', '입니다', '그', '등', '이런', '것 ', '및 ','제', '더', '하다', '19', '종합']
    meaningful_words = [w for w in tokens if not w in stops]
    return ' '.join(meaningful_words)

merge_data['title'] = merge_data['title'].map(remove_stopwords)

# 데이터 분할
X_train, X_val, y_train, y_val = train_test_split(
    merge_data['title'],
    merge_data['topic_idx'],
    test_size = 0.2,
    random_state = 42
)

# vectorizer = TfidfVectorizer()

# 최적값 적용
vectorizer = TfidfVectorizer(max_df=0.8, min_df=1, ngram_range=(1, 1))
X_train_vectorized = vectorizer.fit_transform(X_train)
X_val_vectorized = vectorizer.transform(X_val)

# 모델 파이프라인 생성
# nb_pipeline = make_pipeline(vectorizer, MultinomialNB())

# 하이퍼파라미터 그리드 설정
# nb_param_grid = {
#     'tfidfvectorizer__max_df': [0.8, 0.9, 1.0],
#     'tfidfvectorizer__min_df': [1, 2],
#     'tfidfvectorizer__ngram_range': [(1, 1), (1, 2)],
#     'multinomialnb__alpha': [0.1, 0.5, 1.0]
# }

# GridSearchCV로 하이퍼파라미터 최적화 (Naive Bayes)
# nb_grid_search = GridSearchCV(nb_pipeline, nb_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
# nb_grid_search.fit(X_train, y_train)

# 최적의 하이퍼파라미터 출력
# print("Best parameters for Naive Bayes: ", nb_grid_search.best_params_)
# print("Best cross-validation accuracy for Naive Bayes: {:.2f}".format(nb_grid_search.best_score_))

# 최적값으로 모델학습
model = MultinomialNB(alpha=0.5)
model.fit(X_train_vectorized, y_train)

# 검증 데이터에 대한 예측
y_val_pred = model.predict(X_val_vectorized)

# 모델 평가
accuracy = accuracy_score(y_val, y_val_pred)
classification_rep = classification_report(y_val, y_val_pred, target_names=topic_data['topic'].tolist())

print(f"검증 정확도: {accuracy:.2%}")
print("분류 보고서:")
print(classification_rep)

# 혼동 행렬 생성 및 시각화
conf_matrix = confusion_matrix(y_val, y_val_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=topic_data['topic'].tolist(),
            yticklabels=topic_data['topic'].tolist())
plt.xlabel('예측된 주제')
plt.ylabel('실제 주제')
plt.title('Confusion Matrix Heatmap')
plt.show()

# 주제별 자주 등장하는 단어 분석 및 시각화
feature_names = vectorizer.get_feature_names_out()

for topic_name, topic_group in merge_data.groupby('topic'):
    # 특정 주제에 대한 제목 벡터화
    topic_vectorized = vectorizer.transform(topic_group['title'])

    # 각 단어에 대한 TF-IDF 점수 합산
    word_scores = topic_vectorized.sum(axis=0).A1
    word_scores_df = pd.DataFrame({'term': feature_names, 'score': word_scores})
    word_scores_df = word_scores_df.sort_values(by='score', ascending=False).head(10)

    # 상위 단어에 대한 막대 그래프 시각화
    plt.figure(figsize=(10, 6))
    plt.bar(word_scores_df['term'], word_scores_df['score'])
    plt.title(f'{topic_name} 주제의 주요 단어')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('TF-IDF 점수')
    plt.show()

    # 워드클라우드 생성
    wordcloud = WordCloud(width=800, height=400, background_color='white', font_path=font_path).generate_from_frequencies(
        dict(zip(word_scores_df['term'], word_scores_df['score']))
    )

    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'{topic_name} 주제의 워드클라우드')
    plt.axis('off')
    plt.show()