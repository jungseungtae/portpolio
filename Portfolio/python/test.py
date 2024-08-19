# import requests
# import pandas as pd
# from xml.etree.ElementTree import ElementTree, fromstring
#
# pd.set_option('display.max_columns', None)  # 모든 열을 표시
#
# # 공공데이터 포털 API 키
# api_key = 'RgrQi565jZZmr7M80uvqG9YGaZG34z+mwSutBIsGlWp/8jRKZueDfBL1GckBQl6/j7cFGZIbtmPsbtWIu8Vp0A=='
#
# # API 요청 URL
# url = 'http://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList'
#
# # 요청 파라미터 설정
# params = {
#     'ServiceKey': api_key,
#     'pageNo': 1,
#     'numOfRows': 10,  # 한 페이지에 최대한 많은 데이터를 요청
#     'dataType': 'XML',  # XML 형식으로 데이터 요청
#     'dataCd': 'ASOS',
#     'dateCd': 'DAY',
#     'startDt': '20100701',  # 시작 날짜
#     'endDt': '20100701',  # 종료 날짜
#     'stnIds': '108'
# }
#
#
# def get_asos_data(url, params):
#     response = requests.get(url, params = params)
#     if response.status_code == 200:
#         print(f"API Response: {response.content.decode('utf-8')}")
#         return response.content
#     else:
#         print(f"Error Code: {response.status_code}")
#         return None
#
#
# def parse_xml_to_df(xml_data):
#     tree = ElementTree(fromstring(xml_data))
#     root = tree.getroot()
#     items = root.findall('.//item')
#
#     data = []
#     for item in items:
#         data_dict = {}
#         for child in item:
#             data_dict[child.tag] = child.text
#         data.append(data_dict)
#
#     return pd.DataFrame(data)
#
#
# # XML 데이터 가져오기
# xml_data = get_asos_data(url, params)
#
# if xml_data:
#     # XML 데이터 파싱
#     df = parse_xml_to_df(xml_data)
#
#     # 필요한 열만 선택
#     required_columns = ['tm', 'stnId', 'minTa', 'maxTa', 'hr1MaxIcsrHrmt', 'sumRn',
#                         'hr1MaxRnHrmt', 'hr1MaxRn', 'mi10MaxRnHrmt', 'mi10MaxRn',
#                         'avgTa', 'minTaHrmt', 'maxTaHrmt', 'avgWs']
#
#     # 존재하는 열만 선택하도록 수정
#     available_columns = [col for col in required_columns if col in df.columns]
#     df = df[available_columns]
#
#     # 데이터 타입 변환
#     if 'tm' in df.columns:
#         df['tm'] = pd.to_datetime(df['tm'])
#         for col in ['minTa', 'maxTa', 'hr1MaxRn', 'mi10MaxRn', 'avgTa', 'avgWs']:
#             if col in df.columns:
#                 df[col] = pd.to_numeric(df[col], errors = 'coerce')
#
#         print(df.head())
#     else:
#         print("The 'tm' column is not present in the data.")
#         print(df.columns)
#         print(df)
# else:
#     print("Failed to retrieve data")

import torch

is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')
print(is_cuda)

print('Current cuda decvice is', device)

# print(torch.version.cuda)