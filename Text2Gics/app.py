"""
企业信息转gics行业分类接口
传入打他frame
pd.DataFrame([{'company_name':'上海博高科技有限公司',
                      'brief':'上海博高科技有限公司',
                      'main_business':'从事机电设备技术、机电自动化设备技术、轴承技术领域内的技术咨询、技术开发、技术服务，机电产品、电子元件、计算机、软件及辅助设备、滑动轴承的销售，滑动轴承、油气封的加工、生产。',
                      'business_scope':'上海博高科技有限公司'}],index = [0])
返回行业分类
"""
import os
import traceback
import sys

import pandas as pd

from LSTMBert.model import IndustryClassifierModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from flask import Flask, request
import datetime
import json

import logging
from config.log import logger

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
@app.route('/')
def index():
    return '首页'

@app.route('/text2gics/predict', methods=["GET"])
def predict_gics():
    response = "Fail"
    try:
        df = request.json
        print(df)
        model_path = 'LSTMBert/save/%s/weights/model%08d.h5' % ('r2', 40000)
        predict = IndustryClassifierModel(locals()['r2'], detail=False).predictor(model_path)
        response = predict[0]
        print('ok')
        logger(response)

    except Exception as e:
        now = datetime.datetime.now()
        logger(str(now) + ": " + traceback.format_exc())
    finally:
        print(response)
        return response


if __name__ == '__main__':
    app.run()

    sys.argv = [sys.argv[0]]
    logging.basicConfig(level=logging.INFO,  # 控制台打印的日志级别
                        filename='setting/log.txt',  # 将日志写入log.txt文件中
                        filemode='a',  ##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                        # a是追加模式，默认如果不写的话，就是追加模式
                        format=
                        '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                        # 日志格式
                        )
    app.run(host="0.0.0.0", port=7960, threaded=True)
