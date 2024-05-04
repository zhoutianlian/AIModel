import requests
import pandas as pd
#
# url = 'http://192.168.1.109:7960/val/financing'
# data = {
#     "data": [[{"publish_time": "2015-03-04", "money": "10.9万人民币", "turn": "E轮"},
#               {"publish_time": "2015-09-07", "money": "4.636万人民币", "turn": "D轮"},
#               {"publish_time": "2016-02-04", "money": "5258.54万元", "turn": "C轮"},
#               {"publish_time": "2016-04-08", "money": "1.627亿人民币", "turn": "B轮"},
#               {"publish_time": "2017-09-01", "money": "341.665万美元", "turn": "A+轮"},
#               {"publish_time": "2019-03-31", "money": "658.89万人民币", "turn": "A轮"}]],
#     "if_last": 0
# }
# response = requests.post(url=url, json=data).text
# print(response)

url = 'http://192.168.1.109:7960/val/financing_bydate'
data = {
    "date": 20210510,
    "if_last": 1
}

response = requests.get(url=url, data=data).json()
print(pd.DataFrame(response))

