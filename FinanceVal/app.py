"""
融资估值接口
传入融资数据，调用模型估值
传入所需追溯的天数，从数据库读取对应有融资事件发生的公司返回估值
"""
# -*- coding: utf-8 -*-：
import sys
import os
import traceback

from valuation_by_financing import  ValFinance

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from flask import Flask, request
import datetime
import json
from CONFIG.log import logger
import logging
import pandas as pd
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

@app.route('/')
def index():
    return '首页'
"""
输入参数
是否只返回最后一期估值if_last 默认为1
传入dataframe格式的历史融资记录，参考financing_event的data字段
{
    "data":[[{"publish_time":"2015-03-04","money":"10.9万人民币","turn":"E轮"},
            {"publish_time": "2015-09-07", "money": "4.636万人民币", "turn": "D轮"},
            {"publish_time": "2016-02-04", "money": "5258.54万元", "turn": "C轮"},
            {"publish_time": "2016-04-08", "money": "1.627亿人民币", "turn": "B轮"},
            {"publish_time": "2017-09-01", "money": "341.665万美元", "turn": "A+轮"},
            {"publish_time": "2019-03-31", "money": "658.89万人民币", "turn": "A轮"}]],
    "if_last":0}
"""
@app.route('/val/financing', methods=["POST"])
def val_financing():
    response = "Fail"
    try:
        df = request.json
        logger(df)
        print(df)
        if_last = df['if_last']
        df = pd.DataFrame(df)[['data']]

        vf = ValFinance(target_company=[], n_day=0, date=0)
        ret = vf.pred_flask(data=df,if_last=if_last)
        result = ret.to_dict()
        response = json.dumps(result, ensure_ascii=False)
        print('ok')
        logger(response)

    except Exception as e:
        now = datetime.datetime.now()
        logger(str(now) + ": " + traceback.format_exc())
    finally:
        print(response)
        return response

"""
传入参数
所需估值的日期 20200101
是否只返回最后一期估值if_last 默认为1

"""

@app.route('/val/financing_bydate', methods=["GET"])
def val_financing_bydate():
    response = "Fail"
    try:
        date = request.args.get("date", type=int, default=None)
        if_last = request.args.get("if_last", type=int, default=None)
        vf = ValFinance(target_company=[], n_day=0, date=date)
        df = vf.get_data()
        vf.record_data(df=df, if_last=if_last)
        # ret = vf.pred(df=df,if_last=if_last)
        # result = ret.to_dict()
        # response = json.dumps(result, ensure_ascii=False)

        response = 'Success'
        logger(response)

    except Exception as e:
        now = datetime.datetime.now()
        logger(str(now) + ": " + traceback.format_exc())
    finally:
        print(response)
        return response

if __name__ == '__main__':
    sys.argv = [sys.argv[0]]
    app.run(host="0.0.0.0", port=7960, threaded=True)
    logging.basicConfig(level=logging.INFO,  # 控制台打印的日志级别
                        filename='setting/log.txt',  # 将日志写入log.txt文件中
                        filemode='a',  ##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                        # a是追加模式，默认如果不写的话，就是追加模式
                        format=
                        '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                        # 日志格式
                        )

    # df = pd.DataFrame({
    #     "financing_date": [20190331, 20170901, 20160408, 20160204, 20150907, 20150304],
    #     "financing_amount": [6.588900e+06, 2.206038e+07, 1.627163e+08, 5.258540e+07, 4.636000e+04, 1.090000e+05],
    #     "turn": ["定向增发", "股权转让", "B轮", "股权转让", "股权转让", "定向增发"],
    #     "announce": ["658.89万人民币", "341.665万美元", "1.627亿人民币", "5258.54万元", "4.636万人民币", "10.9万人民币"]
    # })
    #
    # df = pd.DataFrame({
    # "data":[[{"publish_time":"2015-03-04","money":"10.9万人民币","turn":"E轮"},
    #         {"publish_time": "2015-09-07", "money": "4.636万人民币", "turn": "D轮"},
    #         {"publish_time": "2016-02-04", "money": "5258.54万元", "turn": "C轮"},
    #         {"publish_time": "2016-04-08", "money": "1.627亿人民币", "turn": "B轮"},
    #         {"publish_time": "2017-09-01", "money": "341.665万美元", "turn": "A+轮"},
    #         {"publish_time": "2019-03-31", "money": "658.89万人民币", "turn": "A轮"}]]}
    # )
    # vf = ValFinance(target_company=[], n_day=0)
    # vf.get_ex()
    # vf.pred_flask(data=df)

