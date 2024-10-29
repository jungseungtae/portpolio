import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob_function import MLUtils as f

import koreanize_matplotlib
from matplotlib_inline.backend_inline import set_matplotlib_formats
set_matplotlib_formats('retina')

pd.set_option('display.max_columns', None)  # 모든 열을 표시
pd.set_option('display.unicode.east_asian_width', True)  # 한글 출력 시 정렬
pd.set_option('display.width', 1000)        # 콘솔 출력의 너비를 1000으로 설정



### 1. 데이터 불러오기 ###
df = f.load_csv('data/inflearn-event.csv')
# f.eda(df)



### 2. 데이터 전처리 ###
# 중복 제거
df = df.drop_duplicates(['text'], keep='last')
df['origin'] = df['text']

# 소문자 변환
df['text'] = df['text'].str.lower()

# 단어 통일
df["text"] = df["text"].str.replace(
    "python", "파이썬").str.replace(
    "pandas", "판다스").str.replace(
    "javascript", "자바스크립트").str.replace(
    "java", "자바").str.replace(
    "react", "리액트")

# 관심 강의 추출
df["course"] = df["text"].apply(lambda x: x.split("관심강의")[-1])
df["course"] = df["course"].apply(lambda x: x.split("관심 강의")[-1])
df["course"] = df["course"].apply(lambda x: x.split("관심 강좌")[-1])
df["course"] = df["course"].str.replace(":", "", regex=False)
# print(df.head(30))

# 특정 키워드가 들어가는 댓글을 찾음
search_keyword = ['머신러닝', '딥러닝', '파이썬', '판다스', '공공데이터',
                  'django', '크롤링', '시각화', '데이터분석',
                  '웹개발', '엑셀', 'c', '자바', '자바스크립트',
                  'node', 'vue', '리액트']

# for 문을 통해 해당 키워드가 있는지 여부를 True, False값으로 표시하도록 함
for keyword in search_keyword:
    df[keyword] = df["course"].str.contains(keyword)
# print(df.head(30))

df_python = df[df['text'].str.contains('파이썬|공공데이터ㅣ판타스|tensorflow|pytorch|keras')]
# print(df_python.shape)
# print(df_python)

keyword_rank = df[search_keyword].sum().sort_values(ascending=False)
# print(keyword_rank)

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(
    analyzer='word',  # 캐릭터 단위로 벡터화 할 수도 있음
    tokenizer=None,  # 토크나이저를 따로 지정해 줄 수도 있음
    preprocessor=None,  # 전처리 도구
    stop_words=None,  # 불용어 nltk등의 도구를 사용할 수도 있음
    min_df=2,  # 토큰이 나타날 최소 문서 개수로 오타나 자주 나오지 않는 특수한 전문용어 제거에 좋음
    ngram_range=(3, 6),  # BOW의 단위 갯수의 범위를 지정
    max_features=2000  # 만들 피처의 수, 단어의 수
)

feature_vector = vectorizer.fit_transform(df['course'])
vocab = vectorizer.get_feature_names_out()
# print(vocab)

dist = np.sum(feature_vector, axis=0)
df_freq = pd.DataFrame(dist, columns=vocab)
# print(df_freq.head(1))

df_freq_T = df_freq.T.sort_values(by = 0, ascending=False)
# print(df_freq_T.head(30))

df_freq_T = df_freq_T.reset_index()
df_freq_T.columns = ['course', 'freq']
# print(df_freq_T.head())

# 앞에서 4개까지만 단어를 가져와 강의명 중복을 제거
df_freq_T["course_find"] = df_freq_T["course"].apply(lambda x : " ". join(x.split()[:4]))
# print(df_freq_T["course_find"])

# print(df_freq_T.shape)
df_course = df_freq_T.drop_duplicates(['course_find', 'freq'], keep='first')
# print(df_course.shape)

# 강의명 빈도수별로 정리
df_course = df_course.sort_values(by = 'freq', ascending=False)
# print(df_course.head(20))

from sklearn.feature_extraction.text import TfidfTransformer
tfidftrans = TfidfTransformer()

feature_tfidf = tfidftrans.fit_transform(feature_vector)
# print(feature_tfidf)

tfidf_freq = pd.DataFrame(feature_tfidf.toarray(), columns=vocab)
# print(tfidf_freq)

df_tfidf = pd.DataFrame(tfidf_freq.sum())
df_tfidf_top = df_tfidf.sort_values(by = 0, ascending=False)
# print(df_tfidf_top)

from sklearn.cluster import KMeans
from tqdm import trange
inertia = []

start = 10
end = 70

# for i in trange(start, end):
#     kmeans = KMeans(n_clusters=i, random_state=42)
#     kmeans.fit(feature_tfidf)
#     inertia.append(kmeans.inertia_)
#
# plt.plot(range(start, end), inertia)
# plt.title('KMeans 클러스트 수 비교')
# plt.show()

n_clusters = 70
kmeans = KMeans(n_clusters = n_clusters, random_state=42, n_init=10)
kmeans.fit(feature_tfidf)
prediction = kmeans.predict(feature_tfidf)
df['cluster'] = prediction

# print(df['cluster'].value_counts().head(10))

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

b_inertia = []
silhouettes = []

for i in trange(start, end):
    mkmeans = MiniBatchKMeans(n_clusters = i, random_state=42, n_init=3)
    mkmeans.fit(feature_tfidf)
    b_inertia.append(mkmeans.inertia_)
    silhouettes.append(silhouette_score(feature_tfidf, mkmeans.labels_))

plt.plot(range(start, end), b_inertia)
plt.title('MiniBatchKMeans 클러스터 수')

plt.figure(figsize=(15, 4))
plt.title('shilhouette Score')
plt.plot(range(start, end), silhouettes)
plt.xticks(range(start, end))

plt.show()

from yellowbrick.cluster import KElbowVisualizer
KELbowM = KElbowVisualizer(kmeans, k = (start, end))
KELbowM.fit(feature_tfidf.toarray())
KELbowM.show()

mkmeans = MiniBatchKMeans(n_clusters = n_clusters, random_state=42, n_init=3)
mkmeans.fit(feature_tfidf)
prediction = mkmeans.predict(feature_tfidf)
df['bcluster'] = prediction

# print(df['bcluster'].value_counts().head(10))

# print(df.head())


feature_array = feature_vector.toarray()

labels = np.unique(prediction)
df_cluster_score = []
df_cluster = []

for label in labels:
    id_temp = np.where(prediction == label)
    x_means = np.mean(feature_array[id_temp], axis = 0)
    sorted_means = np.argsort(x_means)[::-1][:n_clusters]

    features = vectorizer.get_feature_names_out()
    best_features = [(features[i], x_means[i]) for i in sorted_means]

    df_score = pd.DataFrame(best_features, columns = ['features', 'score'])
    df_cluster_score.append(df_score)

    df_cluster.append(best_features[0])

accuracy = pd.DataFrame(df_cluster,
                        columns=['features', 'score']).sort_values(by = ['features', 'score'],
                                                                   ascending=False)

print(accuracy.shape)

# data_check = df.loc[df['bcluster'] == 32]
# print(data_check)

# print(df.head())
# df.to_excel(r'C:\Users\jstco\OneDrive\바탕 화면\인프런 댓글분석\result3.xlsx', index=False)

### 데이터 시각화
import seaborn as sns
from sklearn.decomposition import PCA

def plot_popular_courses(df):
    plt.figure(figsize=(15, 6))
    df_course = df['course'].value_counts().head(10)
    sns.barplot(x = df_course.values, y = df_course.index)
    plt.title('Top 10 Popular Courses')
    plt.xlabel('Number of Comments')
    plt.ylabel('Course')
    plt.show()

plot_popular_courses(df)

## 언어 인기 순위
def plot_language_popularity(df, search_keyword):
    keyword_rank = df[search_keyword].sum().sort_values(ascending=False)
    plt.figure(figsize = (10, 6))
    sns.barplot(x = keyword_rank.values, y = keyword_rank.index, hue = keyword_rank.index, palette='pastel')
    plt.title('Language Popularity')
    plt.xlabel('Number of Comments')
    plt.ylabel('Language')
    plt.show()

plot_language_popularity(df, search_keyword)

## 군집 형태 시각화
def plot_cluster_shape(df, feature_tfidf, prediction, n_clusters):
    pca = PCA(n_components= 2)
    pca_components = pca.fit_transform(feature_tfidf.toarray())

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(pca_components[:, 0],
                          pca_components[:, 1],
                          c = prediction,
                          cmap='Spectral',
                          s = 50, alpha = 0.7)

    plt.colorbar(scatter, label = 'Cluster')
    plt.title(f'KMeans Clsutering Visualization (n_clusters = {n_clusters})')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.show()

plot_cluster_shape(df, feature_tfidf, prediction, n_clusters)

pca = PCA(n_components=2)
pca.fit(feature_vector.toarray())

components = pca.components_
keywords = vectorizer.get_feature_names_out()

def get_top_keywords(compnents, keywords, n_top = 10):
    top_keywords = []

    for i, compnent in enumerate(compnents):
        sorted_indices = compnent.argsort()[::-1][:n_top]
        top_keywords.append([keywords[j] for j in sorted_indices])
        print(f'Top keywords for PCA compnent {i + 1} : {top_keywords[i]}')

    return top_keywords

top_keywords = get_top_keywords(components, keywords, n_top=10)

# 시각화
def plot_top_keywords(top_keywords, n_components=2):
    for i in range(n_components):
        plt.figure(figsize=(20, 6))
        sns.barplot(x=components[i, :].argsort()[::-1][:10],
                    y=[keywords[j] for j in components[i, :].argsort()[::-1][:10]],
                    hue = [keywords[j] for j in components[i, :].argsort()[::-1][:10]],
                    palette="pastel")
        plt.title(f"Top 10 Keywords for PCA Component {i+1}")
        plt.xlabel("Weight")
        plt.ylabel("Keyword")
        plt.show()

# 키워드 시각화
plot_top_keywords(top_keywords, n_components=2)