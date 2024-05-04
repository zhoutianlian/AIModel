"""
行业分类功能测试
"""
from model import IndustryClassifierModel
import pandas as pd
from data import TEXT_COL
from config import gics_dict
from structure import *


files = ['trainset', 'validset', 'unlabel']

models = {'r2': 40000}

def match_name(code):
    return gics_dict.setdefault(code, None)

output_path = 'data/result.csv'
header = pd.DataFrame([], columns=['gics_code', 'gics_name', 'pred_code', 'pred_name', 'model', 'source',
                                   'company_name', 'brief', 'main_business', 'business_scope'])

def evaluate(d):
    d['pred_code'] = predict(d[TEXT_COL])
    d[['gics_name', 'pred_name']] = d[['gics_code', 'pred_code']].applymap(match_name)
    d = d[['gics_code', 'gics_name', 'pred_code', 'pred_name', 'model', 'source'] + TEXT_COL]
    d.to_csv(output_path, header=0, index=0, mode='a')

total_done = 0
try:
    total_done = pd.read_csv(output_path).shape[0]
except:
    header.to_csv(output_path, index=0)

for model_name, model_version in models.items():
    model_path = 'save/%s/weights/model%08d.h5' % (model_name, model_version)
    print (model_name)
    predict = IndustryClassifierModel(locals()[model_name], detail=False).predictor(model_path)
    for f in files:
        data = pd.read_csv('data/%s.csv' % f)
        count_data = data.shape[0]
        print('Count data in %s = %d' % (f, count_data))
        if count_data > total_done:
            print('Exclude %d data in %s, %d left!' % (total_done, f, count_data - total_done))
            data = data.iloc[total_done:]
            total_done = 0
            data['model'] = '%s_%08d' % (model_name, model_version)
            data['source'] = f

            while len(data) > 1000:
                part_data = data.iloc[:1000]
                data = data.iloc[1000:]
                evaluate(part_data)
            else:
                evaluate(data)

        else:
            total_done -= count_data
            continue

# test_data = pd.read_csv(output_path)
# test_data.to_excel('data/result.xlsx', index=0)

if __name__ =="__main__":

    d = pd.DataFrame([{'company_name':'上海博高科技有限公司',
                      'brief':'上海博高科技有限公司',
                      'main_business':'从事机电设备技术、机电自动化设备技术、轴承技术领域内的技术咨询、技术开发、技术服务，机电产品、电子元件、计算机、软件及辅助设备、滑动轴承的销售，滑动轴承、油气封的加工、生产。',
                      'business_scope':'上海博高科技有限公司'}],index = [0])
    for model_name, model_version in models.items():
        model_path = 'save/%s/weights/model%08d.h5' % (model_name, model_version)
        print (model_name)
        predict = IndustryClassifierModel(locals()[model_name], detail=False).predictor(model_path)
        print(predict(d[TEXT_COL]))
