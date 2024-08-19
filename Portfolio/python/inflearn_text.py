import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob_function import MLUtils as fun

import koreanize_matplotlib
from matplotlib_inline.backend_inline import set_matplotlib_formats
set_matplotlib_formats('retina')

pd.set_option('display.max_columns', None)  # 모든 열을 표시
pd.set_option('display.unicode.east_asian_width', True)  # 한글 출력 시 정렬
pd.set_option('display.width', 1000)        # 콘솔 출력의 너비를 1000으로 설정

df = pd.read_csv('data/inflearn-event.csv')
# fun.eda(df)

df = df.drop_duplicates(['text'], keep = 'last')
df['origin'] = df['text']

df['text'] = df['text'].str.lower()

df["text"] = df["text"].str.replace(
    "python", "파이썬").str.replace(
    "pandas", "판다스").str.replace(
    "javascript", "자바스크립트").str.replace(
    "java", "자바").str.replace(
    "react", "리액트")

df["course"] = df["text"].apply(lambda x: x.split("관심강의")[-1])
df["course"] = df["course"].apply(lambda x: x.split("관심 강의")[-1])
df["course"] = df["course"].apply(lambda x: x.split("관심 강좌")[-1])
df["course"] = df["course"].str.replace(":", "")

# print(df.head())

# 특정 키워드가 들어가는 댓글을 찾음
search_keyword = ['머신러닝', '딥러닝', '파이썬', '판다스', '공공데이터',
                  'django', '크롤링', '시각화', '데이터분석',
                  '웹개발', '엑셀', 'c', '자바', '자바스크립트',
                  'node', 'vue', '리액트']

for keyword in search_keyword:
    df[keyword] = df["course"].str.contains(keyword)

# print(df.head())

df_python = df[df["text"].str.contains("파이썬|공공데이터|판다스")].copy()
# print(df_python)

keyword_mod = df[search_keyword].sum().sort_values(ascending = False)
# print(keyword_mod.sum)

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(
    analyzer = 'word', # 캐릭터 단위로 벡터화 할 수도 있음
    tokenizer = None, # 토크나이저를 따로 지정해 줄 수도 있음
    preprocessor = None, # 전처리 도구
    stop_words = None, # 불용어 nltk등의 도구를 사용할 수도 있음
    min_df = 2, # 토큰이 나타날 최소 문서 개수로 오타나 자주 나오지 않는 특수한 전문용어 제거에 좋음
    ngram_range=(3, 6), # BOW의 단위 갯수의 범위를 지정
    max_features = 2000 # 만들 피처의 수, 단어의 수
)

feature_vector = vectorizer.fit_transform(df['course'])
# print(feature_vector.shape)

vocab = vectorizer.get_feature_names_out()
# print(vocab)

dist = np.sum(feature_vector, axis = 0)
# print(dist)

df_freq = pd.DataFrame(dist, columns = vocab)
# print(df_freq)
# print(df_freq.T.sort_values(by = 0, ascending = False).head(30))

df_freq_T = df_freq.T.reset_index()
df_freq_T.columns = ['course', 'freq']
# print(df_freq_T)

df_freq_T['course_find'] = df_freq_T['course'].apply(lambda x : ' '.join(x.split()[:4]))
# print(df_freq_T)

df_course = df_freq_T.drop_duplicates(['course_find', 'freq'], keep = 'first')
# print(df_course.shape)

df_course = df_course.sort_values(by = 'freq', ascending = False)
# print(df_course)

# df_course.to_pickle('course_freq.pkl')
# df = pd.read_pickle('course_freq.pkl')

from sklearn.feature_extraction.text import TfidfTransformer
tfidftrans = TfidfTransformer(smooth_idf = False)
feature_tfidf = tfidftrans.fit_transform(feature_vector)
# print(feature_tfidf.shape)

tfidf_freq = pd.DataFrame(feature_tfidf.toarray(), columns = vocab)
# print(tfidf_freq.head())

df_tfidf = pd.DataFrame(tfidf_freq.sum())
df_tfidf_top = df_tfidf.sort_values(by = 0, ascending = False)
# print(df_tfidf_top)

from sklearn.cluster import KMeans
from tqdm import trange

inertia = []

start = 10
end = 70

# for i in trange(start, end):
#     kmeans = KMeans(n_clusters = i, random_state = 42)
#     kmeans.fit(feature_tfidf)
#     inertia.append(kmeans.inertia_)

# plt.plot(range(start, end), inertia)
# plt.title('KMeans Cluster')
# plt.show()

n_clusters = 50
kmeans = KMeans(n_clusters = n_clusters, random_state = 42, n_init = 10)
kmeans.fit(feature_tfidf)
prediction = kmeans.predict(feature_tfidf)
df['cluster'] = prediction
# print(df['cluster'].value_counts())

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

b_inertia = []
silhouettes = []

# for i in trange(start, end):
#     mkmeans = MiniBatchKMeans(n_clusters = i, random_state = 42)
#     mkmeans.fit(feature_tfidf)
#     b_inertia.append(mkmeans.inertia_)
#     silhouettes.append(silhouette_score(feature_tfidf, mkmeans.labels_))

# plt.plot(range(start, end), b_inertia)
# plt.title('MiniBatchKMeans Cluster')
# plt.show()

# plt.figure(figsize = (15, 4))
# plt.title('Silhouette Score')
# plt.plot(range(start, end), silhouettes)
# plt.xticks(range(start, end))
# plt.show()

# from yellowbrick.cluster import KElbowVisualizer
#
# KElbowM = KElbowVisualizer(kmeans, k = (start, end))
# KElbowM.fit(feature_tfidf.toarray())
# KElbowM.show()

mkmeans = MiniBatchKMeans(n_clusters = n_clusters, random_state = 42, n_init = 3)
mkmeans.fit(feature_tfidf)
prediction = mkmeans.predict(feature_tfidf)
df['bcluster'] = prediction
# print(df['bcluster'].value_counts())

# search_cluster = df.loc[df['bcluster'] == 20, 'course'].value_counts()
# print(search_cluster)

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

accuracy = pd.DataFrame(df_cluster, columns = ['features', 'score']).sort_values(
    by = ['features', 'score'], ascending = False
)

# print(accuracy)
# df.info()

val = df.loc[df["bcluster"] == 34, ["bcluster", "cluster", "origin", "course"]]
print(val)


### 추가 그래프
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
#
# # 군집 결과를 포함한 데이터프레임이 이미 존재한다고 가정 (df['bcluster'])
# # 각 클러스터의 크기 시각화
# cluster_counts = df['bcluster'].value_counts().sort_index()
#
# # 클러스터 크기 막대 그래프
# plt.figure(figsize=(10, 6))
# cluster_counts.plot(kind='bar')
# plt.title('클러스터 크기')
# plt.xlabel('클러스터 번호')
# plt.ylabel('댓글 수')
# plt.xticks(rotation=0)
# plt.tight_layout()
# plt.show()
#
# # t-SNE를 사용한 데이터 2차원 시각화
# # PCA로 차원 축소 후 t-SNE 적용 (효율성 개선)
# pca = PCA(n_components=50, random_state=42)
# X_pca = pca.fit_transform(feature_tfidf.toarray())
#
# tsne = TSNE(n_components=2, random_state=42, perplexity=30)
# X_tsne = tsne.fit_transform(X_pca)
#
# # t-SNE 시각화
# plt.figure(figsize=(12, 8))
# scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=df['bcluster'], cmap='tab10', alpha=0.7)
# plt.title('t-SNE 시각화')
# plt.xlabel('t-SNE 1')
# plt.ylabel('t-SNE 2')
# plt.colorbar(scatter, ticks=range(n_clusters))
# plt.tight_layout()
# plt.show()
#
# # 클러스터 상위 10개 키워드를 원그래프로 시각화
# top_keywords = keyword_mod.head(10)
#
# plt.figure(figsize=(8, 6))
# top_keywords.plot(kind='pie', autopct='%1.1f%%', startangle=140)
# plt.title('상위 10개 키워드 분포')
# plt.ylabel('')
# plt.tight_layout()
# plt.show()