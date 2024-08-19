import pandas as pd
from sklearn.preprocessing import StandardScaler

file_path = r"C:\Users\jstco\OneDrive\바탕 화면\포트폴리오_강수량\전처리 데이터\csv_data\3. 홍수위험도.csv"

df = pd.read_csv(file_path)
# df.info()

scaler = StandardScaler()
df.iloc[:, 2:] = scaler.fit_transform(df.iloc[:, 2:])

# print(df.head())

df.to_csv(r"C:\Users\jstco\OneDrive\바탕 화면\포트폴리오_강수량\전처리 데이터\csv_data\3. 홍수위험도_scaler.csv")