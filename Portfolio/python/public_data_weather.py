import requests
import pandas as pd


## json
def get_public_data_json(service_key, endpoint, params):
    """
    공공데이터 포털 API를 통해 데이터를 가져오는 함수

    Parameters:
    - service_key: 공공데이터 포털에서 발급받은 서비스 키
    - endpoint: API 엔드포인트 URL
    - params: API 요청 매개변수 딕셔너리

    Returns:
    - DataFrame: 응답 데이터를 데이터프레임으로 반환
    """
    params['serviceKey'] = service_key
    response = requests.get(endpoint, params = params)

    if response.status_code == 200:
        data = response.json()
        # 데이터 파싱 및 DataFrame 변환
        if 'response' in data and 'body' in data['response'] and 'items' in data['response']['body']:
            items = data['response']['body']['items']['item']
            df = pd.DataFrame(items)
            return df
        else:
            print("응답 데이터에 예상된 구조가 없습니다.")
            return None
    else:
        print(f"API 요청 실패: {response.status_code}")
        return None


# 예제 사용
service_key = 'RgrQi565jZZmr7M80uvqG9YGaZG34z%2BmwSutBIsGlWp%2F8jRKZueDfBL1GckBQl6%2Fj7cFGZIbtmPsbtWIu8Vp0A%3D%3D'  # 공공데이터 포털에서 발급받은 서비스 키
endpoint = 'http://openapi.molit.go.kr:8081/OpenAPI_ToolInstallPackage/service/rest/RTMSOBJSvc/getRTMSDataSvcAptRent?_wadl&type=xml'  # 사용하려는 API 엔드포인트

params = {
    'pageNo': '1',
    'numOfRows': '10',
    'type': 'XML'
}

# df = get_public_data_json(service_key, endpoint, params)

# if df is not None:
#     print(df.head())
# else:
#     print("데이터를 가져오지 못했습니다.")


################################################################

import requests
from bs4 import BeautifulSoup
import pandas as pd


def get_public_data_xml(service_key, endpoint, params):
    """
    공공데이터 포털 API를 통해 XML 데이터를 가져오는 함수

    Parameters:
    - service_key: 공공데이터 포털에서 발급받은 서비스 키
    - endpoint: API 엔드포인트 URL
    - params: API 요청 매개변수 딕셔너리

    Returns:
    - DataFrame: 응답 데이터를 데이터프레임으로 반환
    """
    params['serviceKey'] = service_key
    response = requests.get(endpoint, params = params)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'xml')

        # 데이터 파싱
        items = soup.find_all('item')
        data = []
        for item in items:
            row = {}
            for child in item.children:
                if child.name is not None:
                    row[child.name] = child.text
            data.append(row)

        df = pd.DataFrame(data)
        return df
    else:
        print(f"API 요청 실패: {response.status_code}")
        return None


# 예제 사용
service_key = 'RgrQi565jZZmr7M80uvqG9YGaZG34z%2BmwSutBIsGlWp%2F8jRKZueDfBL1GckBQl6%2Fj7cFGZIbtmPsbtWIu8Vp0A%3D%3D'  # 공공데이터 포털에서 발급받은 서비스 키
endpoint = 'http://openapi.molit.go.kr:8081/OpenAPI_ToolInstallPackage/service/rest/RTMSOBJSvc/getRTMSDataSvcAptRent?_wadl&type=xml'  # 사용하려는 API 엔드포인트

params = {
    # 'pageNo': '1',
    # 'numOfRows': '10'
    # 'LAWD_CD': '11110',
    # 'DEAL_YMD': '201512'
}

# df = get_public_data_xml(service_key, endpoint, params)
# if df is not None:
#     print(df.head())
# else:
#     print("데이터를 가져오지 못했습니다.")


#################################################
import requests
import xml.etree.ElementTree as ET
import pandas as pd

def get_items(response):
    try:
        root = ET.fromstring(response.content)
        item_list = []
        body = root.find('body')
        if body is None:
            raise ValueError("Response does not contain 'body'")
        items = body.find('items')
        if items is None:
            raise ValueError("Response does not contain 'items'")
        for child in items.findall('item'):
            elements = child.findall('*')
            data = {}
            for element in elements:
                tag = element.tag.strip()
                text = element.text.strip() if element.text else ''
                data[tag] = text
            item_list.append(data)
        return item_list
    except ET.ParseError:
        print("Failed to parse XML")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

# 예제 사용
service_key = service_key  # 공공데이터 포털에서 발급받은 서비스 키
url = "http://openapi.molit.go.kr:8081/OpenAPI_ToolInstallPackage/service/rest/RTMSOBJSvc/getRTMSDataSvcAptRent"
base_date = "202001"
gu_code = '11215'  # 법정동 코드 5자리

payload = "LAWD_CD=" + gu_code + "&" + \
          "DEAL_YMD=" + base_date + "&" + \
          "serviceKey=" + service_key

res = requests.get(url + "?" + payload)

# 응답 내용 출력
# print(res.content)

if res.status_code == 200:
    items_list = get_items(res)
    if items_list:
        items = pd.DataFrame(items_list)
        print(items.head())
    else:
        print("No data available")
else:
    print(f"Request failed with status code: {res.status_code}")