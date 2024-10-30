import re
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns
from tqdm import tqdm
import koreanize_matplotlib
from transformers import BertTokenizer, BertModel
import os
import pickle
from collections import Counter

is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')
print(is_cuda)

print('Current cuda decvice is', device)

# 한글 폰트 설정
from matplotlib_inline.backend_inline import set_matplotlib_formats
set_matplotlib_formats('retina')

pd.set_option('display.max_columns', None)
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.width', 1000)



'''''''''''''''

1. 데이터 불러오기

'''''''''''''''
df = pd.read_csv('data/inflearn-event.csv')
# print(df.head())




'''''''''''''''

2. 데이터 전처리

'''''''''''''''
# 중복 제거
df = df.drop_duplicates(['text'], keep='last')
df['origin'] = df['text']

# 소문자 변환 및 특수문자 제거, 불용어 제거
def preprocess_text(text):
    text = text.lower()  # 소문자 변환
    text = re.sub(r'[^가-힣a-zA-Z0-9\s]', '', text)  # 특수문자 제거
    return text

df['text'] = df['text'].apply(preprocess_text)

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

# bert 모델 및 토크나이저
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', clean_up_tokenization_spaces=True)
model = BertModel.from_pretrained('bert-base-multilingual-cased')

# 결과 저장 경로
bert_embedding_file = "data/bert_embeddings.pkl"
cluster_prediction_file = "data/cluster_predictions.pkl"

# bert 임베딩 생성 함수
def get_bert_embeddings(text_list):
    inputs = tokenizer(text_list, return_tensors = 'pt', padding = True, truncation = True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # [CLS] 토큰의 임베딩 값 (문장 임베딩으로 사용)
    cls_embeddings = outputs.last_hidden_state[:, 0, :]
    return cls_embeddings.numpy()


# bert 임베딩 처리
if os.path.exists(bert_embedding_file):
    # 저장된 BERT 임베딩 데이터를 불러옴
    with open(bert_embedding_file, 'rb') as f:
        bert_embeddings = pickle.load(f)
    print("BERT 임베딩 데이터를 불러왔습니다.")
else:
    # BERT 임베딩을 생성하여 저장
    texts = df['course'].tolist()
    bert_embeddings = []

    for text in tqdm(texts):
        bert_embedding = get_bert_embeddings([text])
        bert_embeddings.append(bert_embedding)

    bert_embeddings = np.vstack(bert_embeddings)

    # 임베딩 데이터를 pkl 파일로 저장
    with open(bert_embedding_file, 'wb') as f:
        pickle.dump(bert_embeddings, f)
    print("BERT 임베딩 데이터를 저장했습니다.")

# 최적의 클러스터 수 찾기 (엘보우 방법)
def find_optimal_clusters(data, max_k):
    iters = range(1, max_k + 1)
    inertia = []

    for k in iters:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)  # n_init 명시적으로 설정
        kmeans.fit(data)
        inertia.append(kmeans.inertia_)

    # 엘보우 방법 그래프 그리기
    plt.figure(figsize=(10, 6))
    plt.plot(iters, inertia, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal Clusters')
    plt.show()

    # 최적 클러스터 수 추천 (관성이 급격히 줄어드는 지점)
    # 변화율을 계산하여 가장 급격히 변하는 구간을 최적의 클러스터 수로 추천
    diffs = np.diff(inertia)  # inertia 변화량 계산
    diff_ratios = diffs[1:] / diffs[:-1]  # 변화량 비율 계산
    optimal_k = np.argmin(diff_ratios) + 2  # 인덱스에 2를 더해 클러스터 수를 보정

    print(f"Recommended optimal number of clusters: {optimal_k}")
    return optimal_k

# 최적 클러스터 수 찾기
optimal_clusters = find_optimal_clusters(bert_embeddings, max_k=70)   # 17




'''''''''''''''

3. 모델 적용

'''''''''''''''
n_clusters = optimal_clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(bert_embeddings)





'''''''''''''''

4. 데이터 시각화

'''''''''''''''
# # 클러스터링 결과 시각화
# def plot_cluster_shape(df, bert_embeddings, prediction, n_clusters):
#     pca = PCA(n_components=2)
#     pca_components = pca.fit_transform(bert_embeddings)
#
#     plt.figure(figsize=(12, 8))
#     scatter = plt.scatter(pca_components[:, 0], pca_components[:, 1],
#                           c=prediction, cmap='Spectral', s=50, alpha=0.7)
#     plt.colorbar(scatter, label='Cluster')
#     plt.title(f'BERT Embedding Clustering Visualization (n_clusters = {n_clusters})')
#     plt.xlabel('PCA Component 1')
#     plt.ylabel('PCA Component 2')
#     plt.show()
#
# plot_cluster_shape(df, bert_embeddings, df['cluster'], optimal_clusters)
#
# # 클러스터 번호가 부여된 데이터 확인
# # print(df[['course', 'cluster']].head(10))
#
# # 결과 저장 (엑셀 파일로 저장)
# # output_file = "course_clusters3.xlsx"
# # df.to_excel(output_file, index=False)
# #
# # print(f"DataFrame saved to {output_file}")
#
# search_keyword = ['머신러닝', '딥러닝', '파이썬', '판다스', '공공데이터', 'django', '크롤링', '시각화', '데이터분석', '웹개발', '엑셀', 'c', '자바', '자바스크립트', 'node', 'vue', '리액트']
#
# # 자주 등장하는 언어 순위
# def count_keywords(df, search_keyword):
#     keyword_count = {keyword: 0 for keyword in search_keyword}
#
#     for course in df['course']:
#         for keyword in search_keyword:
#             if keyword in course:
#                 keyword_count[keyword] += 1
#                 break  # 첫 번째 매칭된 키워드만 카운트
#
#     return keyword_count
#
# def plot_keyword_count(df, search_keyword):
#     keyword_count = count_keywords(df, search_keyword)
#     keyword_rank = pd.Series(keyword_count).sort_values(ascending=False)
#
#     # 시각화
#     plt.figure(figsize=(10, 6))
#     sns.barplot(x=keyword_rank.values, y=keyword_rank.index, palette='pastel')
#     plt.title('Top 10 Languages Mentioned')
#     plt.xlabel('Number of Mentions')
#     plt.ylabel('Language')
#     plt.show()
#
# plot_keyword_count(df, search_keyword)
#
# # 각 군집별 데이터 수 시각화
# def plot_cluster_distribution(df):
#     plt.figure(figsize=(10, 6))
#     sns.countplot(x=df['cluster'], palette='coolwarm')
#     plt.title(f'Number of Data Points in Each Cluster (n_clusters = 17)')
#     plt.xlabel('Cluster')
#     plt.ylabel('Number of Data Points')
#     plt.show()
#
# plot_cluster_distribution(df)
#
# # 각 군집별 자주 나오는 단어 출력
# def get_top_keywords_per_cluster(df, n_clusters, top_n=5):
#     cluster_keywords = {}
#
#     for cluster in range(n_clusters):
#         # 각 클러스터에 속한 데이터를 필터링
#         cluster_data = df[df['cluster'] == cluster]
#
#         # 'course' 열에서 모든 단어 추출
#         all_keywords = ' '.join(cluster_data['course']).split()
#
#         # 빈도가 높은 상위 키워드 추출
#         keyword_counts = Counter(all_keywords)
#         top_keywords = keyword_counts.most_common(top_n)
#
#         cluster_keywords[cluster] = top_keywords
#
#     return cluster_keywords
#
# def print_top_keywords_per_cluster(df, n_clusters, top_n=5):
#     cluster_keywords = get_top_keywords_per_cluster(df, n_clusters, top_n)
#
#     for cluster, keywords in cluster_keywords.items():
#         print(f"Cluster {cluster}: {keywords}")
#
# print_top_keywords_per_cluster(df, 17)



'''''''''''''''

5. 모델 평가

'''''''''''''''
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# 1. 실루엣 스코어 계산
silhouette_avg = silhouette_score(bert_embeddings, df['cluster'])
print(f"Silhouette Score: {silhouette_avg:.4f}")

# 2. Calinski-Harabasz Index 계산
calinski_harabasz = calinski_harabasz_score(bert_embeddings, df['cluster'])
print(f"Calinski-Harabasz Index: {calinski_harabasz:.4f}")

# 3. Davies-Bouldin Index 계산
davies_bouldin = davies_bouldin_score(bert_embeddings, df['cluster'])
print(f"Davies-Bouldin Index: {davies_bouldin:.4f}")

'''
평가 방법 설명:
Silhouette Score : 1에 가까울수록 군집이 잘 형성되었다는 의미입니다.

Calinski-Harabasz Index : 
값이 클수록 군집이 더 잘 나뉘어져 있다는 것을 의미합니다. 군집 내 분산이 작고, 군집 간 분산이 클수록 이 값이 커집니다.

Davies-Bouldin Index : 값이 작을수록 군집이 잘 형성되었다는 것을 의미합니다. 군집 내 분산이 작고, 군집 간 분산이 클수록 좋습니다.

Silhouette Score: 0.0936
Calinski-Harabasz Index: 88.3420
Davies-Bouldin Index: 2.4292
'''

# 실루엣 스코어 시각화 (선택 사항):
import matplotlib.cm as cm
from sklearn.metrics import silhouette_samples
def plot_silhouette_score(df, bert_embeddings, n_clusters):
    fig, ax = plt.subplots(figsize=(10, 6))

    # 각 샘플에 대한 실루엣 스코어 계산
    silhouette_values = silhouette_samples(bert_embeddings, df['cluster'])

    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = silhouette_values[df['cluster'] == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.Spectral(float(i) / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, ith_cluster_silhouette_values,
                         facecolor=color, edgecolor=color, alpha=0.7)

        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax.set_title("Silhouette Plot for the Various Clusters")
    ax.set_xlabel("Silhouette Coefficient")
    ax.set_ylabel("Cluster Label")
    plt.axvline(x=silhouette_avg, color="red", linestyle="--")
    plt.show()


# 실루엣 스코어 시각화
plot_silhouette_score(df, bert_embeddings, n_clusters)