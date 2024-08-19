import pandas as pd
import numpy as np
import glob_function as gf
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib_inline.backend_inline import set_matplotlib_formats
import koreanize_matplotlib

set_matplotlib_formats('retina')

### 1. 데이터 불러오기 ###
file_path = r'C:\Users\jstco\Downloads\pytextbook-main\pytextbook-main\data\klue'
train_data = file_path + '/train_data.csv'
test_data = file_path + '/test_data.csv'
topic_data = file_path + '/topic_dict.csv'

train = gf.MLUtils.load_csv(train_data)
test = gf.MLUtils.load_csv(test_data)
topic = gf.MLUtils.load_csv(topic_data)
# print(train.shape, test.shape)
# print(train.head())

raw = pd.concat([train, test])

df = raw.merge(topic, how = 'left')

### 2. 데이터 탐색 ###
# gf.MLUtils.eda(train)
# gf.MLUtils.eda(test)
# gf.MLUtils.eda(topic)

topic_mod = df['topic_idx'].value_counts()
# print(topic_mod)

# sns.countplot(data = df, x = 'topic')
# plt.show()


### 3. 데이터 전처리 ###
df['len'] = df['title'].apply(lambda x: len(x))
df['word_count'] = df['title'].apply(lambda x: len(x.split()))
df['unique_word_count'] = df['title'].apply(lambda x: len(set(x.split())))

# print(df.head())

df['title'] = df['title'].str.replace('[0-9]', '', regex = True)
df['title'] = df['title'].str.lower()

from konlpy.tag import Okt
okt = Okt()

## 조사, 어미, 구두점 제거
def okt_clean(text):
    clean_text = []
    for word in okt.pos(text, stem = True):
        if word[1] not in ['Josa', 'Eomi', 'Punctuation']:
            clean_text.append(word[0])

    return ' '.join(clean_text)

from tqdm import tqdm
tqdm.pandas()

# train['title'] = train['title'].progress_map(okt_clean)
# test['title'] = test['title'].progress_map(okt_clean)

# train.to_pickle('train_processed.pkl')
# test.to_pickle('test_processed.pkl')

train = pd.read_pickle('train_processed.pkl')
test = pd.read_pickle('test_processed.pkl')

# print(train.head())
# print(test.head())

def remove_stopwords(text):
    tokens = text.split(' ')
    stops = ['합니다', '하는', '할', '하고', '한다',
             '그리고', '입니다', '그 ', ' 등', '이런', ' 것 ', ' 및 ',' 제 ', ' 더 ']
    meaningful_words = [w for w in tokens if not w in stops]
    return ' '.join(meaningful_words)

df['title'] = df['title'].map(remove_stopwords)

label_name = 'topic_idx'

## 데이터 분리
train = df[df[label_name].notnull()]
test = df[df[label_name].isnull()]
# print(train.shape, test.shape)

X_train = train['title']
X_test = test['title']

y_train = train[label_name]

### 4. 벡터화 ###
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vect = TfidfVectorizer(tokenizer = None,
                             ngram_range = (1, 2),
                             min_df = 3,
                             max_df = 0.95)
tfidf_vect.fit(X_train)

train_feature_tfidf = tfidf_vect.transform(X_train)
test_feature_tfidf = tfidf_vect.transform(X_test)
# print(train_feature_tfidf)

vocab = tfidf_vect.get_feature_names_out()
# print(vocab[:10])

dist = np.sum(train_feature_tfidf, axis = 0)
vocab_count = pd.DataFrame(dist, columns = vocab)

# vocab_count.T[0].sort_values(ascending = False).head(50).plot.bar(figsize = (15, 10))
# plt.show()


### 5. 모델적용 ###
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state = 42, n_jobs = -1)

## 훈련 데이터 예측 후 검증
from sklearn.model_selection import cross_val_predict
# y_pred = cross_val_predict(model, train_feature_tfidf, y_train, cv = 3, verbose = 1, n_jobs = -1)
# print(y_pred)
# y_pred_df = pd.DataFrame(y_pred, columns = ['pred'])
# y_pred_df.to_pickle('y_pred_processed.pkl')

y_pred = pd.read_pickle('y_pred_processed.pkl')
# print(type(y_pred))
# y_pred.info()
y_pred_series = y_pred['pred']

# print(y_pred_series.info())
# print(y_train.info())

# print(y_pred_series.head())
# print(y_train.head())

valid_accuracy = (y_pred_series == y_train).mean()
# print(valid_accuracy)

df_accuracy = pd.DataFrame({'pred' : y_pred_series, 'train' : y_train})
# print(df_accuracy)

df_accuracy['accuracy'] = (y_pred_series == y_train)
# print(df_accuracy.head())

grouped_accuracy = df_accuracy.groupby(['train'])['accuracy'].mean()
# print(grouped_accuracy)

model.fit(train_feature_tfidf, y_train)

y_predict = model.predict(test_feature_tfidf)

sub = file_path + '/sample_submission.csv'
submit = gf.MLUtils.load_csv(sub)
# print(submit.head())

submit['pred_idx'] = y_predict
submit.to_csv('result.csv', index = False)

result =  './result.csv'
pred = gf.MLUtils.load_csv(result)
# print(pred)