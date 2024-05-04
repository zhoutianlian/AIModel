"""
测试入口

调用模型
依据要求实现融资估值
输入参数
目标企业
目标日期
向前追溯天数


"""
import csv
import os
import re
import pymysql
import datetime
import pandas as pd
import pymongo
from CONFIG.Config import config
from CONFIG.globalENV import global_var
from ValuationRecord.SaveEnterprise import save
import numpy as np
import time
import matplotlib.pyplot as plt
from pymongo import MongoClient
from keras.models import load_model
import pandas as pd
from functools import reduce
import warnings
from utils import to_describe, split_round, determine_round, determine_currency, determine_value_mag_real, to_exact, \
    to_about, weighted_mse, weighted_mae, get_n_day_before, read_mongo, save_mongo, to_data

warnings.filterwarnings("ignore")
# linearregression, predict the future value
# calculate confidence interval
from keras.preprocessing.sequence import pad_sequences



class ValFinance():

    def __init__(self,target_company,n_day,date):
        self.n_day = n_day
        self.date = date
        self.out_host = config['DEFAULT']['out_mongo_host']
        self.out_username = config['DEFAULT']['out_mongo_user']
        self.out_password = config['DEFAULT']['out_mongo_password']
        self.out_port = int(config['DEFAULT']['out_mongo_port'])


        self.target_company = target_company
        self.model_path = os.path.join(os.getcwd(), 'model', 'lstm.h5')
        self.lstm = load_model(self.model_path,custom_objects={'weighted_mse': weighted_mse, 'weighted_mae': weighted_mae})
        self.ex_usd = self.get_ex()

    # clean financing data
    # transfer financing amount per time to accumulative financin amount
    # get the v*4
    # return dataframe with company name and v*4
    """
    读取融资数据
    """
    def get_data(self):
        if self.target_company:
            l_or = []
            for i in self.target_company:
                l_or.append({'company_name': i})
            df = read_mongo(db_name='spider_origin', collection_name='financing_event',
                                 conditions={'$or': l_or},
                                 query=[], sort_item='', n=0)
        elif self.n_day>0:
            df = read_mongo(db_name='spider_origin', collection_name='financing_event',
                                 conditions={'last_financing_date': {'$gte': get_n_day_before(n=self.n_day)}},
                                 query=[], sort_item='', n=0)
        else:
            df = read_mongo(db_name='spider_origin', collection_name='financing_event',
                                 conditions={'last_financing_date':self.date},
                            query=[], sort_item='',n=0)

        # df = self.read_mongo(db_name='spider_origin', collection_name='financing_event', conditions={},
        #                 query=[], sort_item='',n=0)
        # start = get_n_day_before(n=n_day)
        # df = df[df['last_financing_date'] > start]
        print(df)
        df = df.dropna(subset=['company_name'])
        return df
    def get_ex(self):
        ex_usd = read_mongo(db_name='modgo_quant', collection_name='fxTradeData',
                            conditions={'$and':[{'trade_date':{'$gt':get_n_day_before(n = 90)}},{'currency_pair':'USD/CNY'}]}, query=[], sort_item='',
                   n=0)
        ex_usd = ex_usd['fx_rate'].mean()
        return ex_usd
    """
    数据清洗
    """
    def clean_data(self,x):
        df_ret = pd.DataFrame(columns=['financing_date','turn','announce'])
        try:
            for i in range(len(x['data'].values[0])):
                data = x['data'].values[0][i]
                date = pd.to_datetime(data['publish_time'])
                df_i = pd.DataFrame({'financing_date': [date],
                                     'turn': [data['turn']], 'announce': [data['money']]})
                df_ret = df_ret.append(df_i, ignore_index=True)
        except:
            for i in range(len(x['data'])):
                data = x['data'][i]
                date = pd.to_datetime(data['publish_time'])
                df_i = pd.DataFrame({'financing_date': [date],
                                     'turn': [data['turn']], 'announce': [data['money']]})
                df_ret = df_ret.append(df_i, ignore_index=True)

        return df_ret

    def insert_valuation_record_sql(self,data):
        db = pymysql.Connection(
            host=config['DEFAULT']['mysql_host'],
            port=int(config['DEFAULT']['mysql_port']),
            user=config['DEFAULT']['mysql_user'],
            password=config['DEFAULT']['mysql_password'],
            database='rdt_fintech'
        )
        cursor = db.cursor()
        name = data['company_name']
        print(name)
        e_id = save(name=name)
        v_id = 0
        print('e_id', e_id)
        if e_id:
            sql_insert = """
            INSERT INTO t_valuating_record (activity_id, c_time, channel_id, currency_id, 
            enterprise_id, source_id, source_table, user_id, 
            val_accuracy,val_inputmethod,val_ip,val_terminal,
            val_type,val_valid,valuation_failure,val_source_id, enterprise_name) 
            VALUES (%s, %s, %s, %s, %s,
            %s, %s, %s, %s, 
            %s,%s,%s,%s,
            %s,%s,%s,%s)
            """

            val_acc = float(int(data['financing_date'].replace('-','')) - 20000000)
            now = datetime.datetime.now()
            # now = now.strftime("%Y-%m-%d %H:%M:%S")

            values = (0, now, 0, 1,
                      str(e_id), 0, 0, '60',
                      val_acc, 90, '192.168.1.144', 6,
                      10, 1, 0, 0, name)

            try:
                if cursor.execute(sql_insert, values):
                    print("successful")

                    v_id = db.insert_id()
                    print(v_id)
                    db.commit()

            except Exception as e:
                print(e)
                cursor = db.cursor()
                cursor.execute(sql_insert, values)
                print("Failed")
                db.rollback()
        return v_id

    def insert_valuation_record_mongo(self,data, vid):
        now = datetime.datetime.now()
        now = now.strftime("%Y-%m-%d %H:%M:%S")
        # dict_v = {
        #     'syn':[data['v_min'],data['v_avg'],data['v_max']]
        # }
        # df = pd.DataFrame({'val': [dict_v], '_id': [vid],'time':[now]})
        dict_v = {
            'syn': [data['next_val_lower'], data['next_val'], data['next_val_upper']]
        }
        df = pd.DataFrame({'val': [dict_v], '_id': [vid], 'time': [data['financing_date']]})
        try:

            save_mongo(df_data=df, db_name='rdt_fintech', tb_name='ValuationResult', if_del=0)
            print('successful mongo')
        except Exception as e:
            print(e)

    """
    如果感觉估值准确
    模拟客户真实估值的情况
    向sql mongo同时插入估值记录以及估值结果
    """
    # 生成估值记录和估值记录结果方法
    def record_data(self, df, if_last=1):
        df_pred = self.pred_many(df=df, if_last=if_last)
        for i, r in df_pred.iterrows():
            vid = self.insert_valuation_record_sql(data=r)
            if vid>0:
                self.insert_valuation_record_mongo(data=r, vid=vid)
            else:
                print('no vid')


    def update_to_mongo(self,data, v_id):

        mongo_uri = 'mongodb://%s:%s@%s:%s/?authSource=%s' % \
                    (self.out_username, self.out_password, self.out_host, self.out_port,'rdt_fintech')
        conn = MongoClient(mongo_uri)
        db = conn['rdt_fintech']
        collection = db['ValuationResult']
        myquery = {'_id': v_id}
        dict_v = {
            'syn':[data['v_min'],data['v_avg'],data['v_max']]
        }
        value = {'val': dict_v}
        collection.update_one(myquery, {"$set": value})
    """
    如发现已插入的估值结果有问题，批量修改
    """
    def amend_data(self):
        df_pred = self.predict_value()

        db = pymysql.Connection(
            host=config['DEFAULT']['mysql_host'],
            port=int(config['DEFAULT']['mysql_port']),
            user=config['DEFAULT']['mysql_user'],
            password=config['DEFAULT']['mysql_password'],
            database='rdt_fintech'
        )

        for i, r in df_pred.iterrows():
            cursor = db.cursor()
            e_id = save(name=r['company_name'])
            search = 'select id from t_valuating_record where enterprise_id = %s and val_type = 10'
            if cursor.execute(search, e_id):
                data = cursor.fetchall()
                for d in data:
                    self.update_to_mongo(data = r, v_id = d[0])

    """
    针对多家公司。批量估值
    """
    def pred_many(self,df,if_last= 1):

        data = pd.DataFrame()
        for i, x in df.iterrows():
            name = x['company_name']
            x = self.clean_data(x)
            x['company_name'] = name
            data = data.append(x)
        data['financing_date'] = data['financing_date'].apply(pd.Timestamp)
        data = data.sort_values(by=['company_name', 'financing_date'], ignore_index=True)
        data = data.groupby('company_name', as_index=True).apply(to_data)

        # if if_add:
        #     flag = 'a'
        # else:
        #     path = os.path.join(os.getcwd(),'output')
        #     for root, dirs, files in os.walk(path):
        #         res = files
        #     for i in res:
        #         if os.path.exists(os.path.join(os.getcwd(),'output',i)):  # 如果文件存在
        #             # 删除文件，可使用以下两种方法。
        #             os.remove(os.path.join(os.getcwd(),'output',i))
        #             # os.unlink(path)
        #         else:
        #             print('no such file:%s' % i)  # 则返回文件不存在
        #     flag = 'w'
        res = pd.DataFrame()
        for i in data.index.levels[0]:
            try:
                d = data.loc[i]
                if len(d) >= 3:
                    x_val = list()
                    x_lab = list()
                    index = list()
                    if if_last:
                        x_val.append(d[['interval', 'val', 'mag','currency']])
                        x_lab.append(d[['round', 'expr']])
                        d = d.iloc[[-1]]
                    else:
                        for j in range(len(d)):
                            x_val.append(d.iloc[:j + 1][['interval', 'val', 'mag','currency']])
                            x_lab.append(d.iloc[:j + 1][['round', 'expr']])
                            index.append(d.iloc[j]['financing_date'])
                    x_val = pad_sequences(x_val, 10, float)
                    x_lab = pad_sequences(x_lab, 10, float)

                    y = self.lstm.predict([x_val, x_lab])
                    val = pd.DataFrame(y[:, 0, :], columns=['amount', 'post_val', 'next_val'], index=d.index)
                    val = val.applymap(lambda x: pow(10, x))
                    sig = pd.DataFrame(y[:, 1, :], columns=['amount_sig', 'post_val_sig', 'next_val_sig'], index=d.index)
                    sig = sig.applymap(lambda x: pow(10, x))
                    output = pd.concat([d, val, sig], axis=1)
                    output['pct'] = output.eval('amount / post_val')
                    output['amount_lower'] = output.eval('amount / amount_sig')
                    output['amount_upper'] = output.eval('amount * amount_sig')
                    output['post_val_lower'] = output.eval('post_val / post_val_sig')
                    output['post_val_upper'] = output.eval('post_val * post_val_sig')
                    output['next_val_lower'] = output.eval('next_val / next_val_sig')
                    output['next_val_upper'] = output.eval('next_val * next_val_sig')
                    output = output.round(2)
                    res = res.append(output)
                    # output.to_csv('./output/result.csv', header=(flag == 'w'), mode=flag, index=False, encoding='gbk')
                    #
                    # brief['post_val'] = brief['post_val'].apply(lambda x: round(x / 1e4, 2))
                    # brief['amount'] = brief['amount'].apply(lambda x: round(x / 1e4, 2))
                    # brief['est'] = brief['est'].apply(lambda x: round(x / 1e8, 2))
                    # brief['interval'] = brief['interval'].apply(lambda x: round(x, 2))
                    # brief = brief[['financing_date','turn','announce',
                    #     'post_val','company_name','interval','amount','est']]
                    # brief = brief.rename(columns={
                    #     'financing_date': '融资日期',
                    #     'turn': '融资轮次',
                    #     'announce': '披露金额',
                    #     'post_val': '融资估值（亿）',
                    #     'company_name': '公司名字',
                    #     'interval': '估值间隔（年）',
                    #     'amount': '估计融资金额（亿）',
                    #     'est':'披露金额（量化）'
                    # })
                    # brief.to_csv('./output/brief.csv', header=(flag == 'w'), mode=flag, index=False, encoding='gbk')
                    # flag = 'a'
            except Exception as e:
                print(i, e)
                continue
        res = res[['company_name', 'financing_date', 'next_val', 'next_val_lower', 'next_val_upper']]
        res['financing_date'] = res['financing_date'].apply(lambda x: x.strftime("%Y-%m-%d"))
        return res
    """
    针对某一家公司估值
    """
    def pred_flask(self,data,if_last=1):

        data = self.clean_data(data)
        data['financing_date'] = data['financing_date'].apply(pd.Timestamp)
        data = data.sort_values(by=['financing_date'], ignore_index=True)
        data = self.to_data(data)

        if 100 in data['round'].values:
            v = data.query('round == 100')['est'].values[0]
            d = data.query('round == 100')['financing_date']
            res = pd.DataFrame(
            {'v_growth': [v],
             'v_avg': [v], 'v_min': [v*0.8],
             'v_max': [v*1.2],'financing_date':[d]})
            return res
        if if_last:
            d = data
            x_val = [d[['interval', 'val', 'mag', 'currency']].values]
            x_lab = [d[['round', 'expr']].values]
            index = [d['financing_date']]

            x_val = pad_sequences(x_val, 10, float)
            x_lab = pad_sequences(x_lab, 10, int)
            y = self.lstm.predict([x_val, x_lab])
            d = data.iloc[[-1]]


        else:
            res = pd.DataFrame()
            x_val = list()
            x_lab = list()
            index = list()
            for i in range(len(data)):

                d = data.iloc[:i + 1]
                x_val.append(d[['interval', 'val', 'mag', 'currency']])
                x_lab.append(d[['round', 'expr']])
                index.append(data.iloc[i]['financing_date'])
            x_val = pad_sequences(x_val, 10, float)
            x_lab = pad_sequences(x_lab, 10, int)
            y = self.lstm.predict([x_val, x_lab])


        val = pd.DataFrame(y[:,0,:], columns=['amount', 'post_val', 'next_val'], index=d.index)
        val = val.applymap(lambda x: pow(10, x))
        sig = pd.DataFrame(y[:, 1, :], columns=['amount_sig', 'post_val_sig', 'next_val_sig'], index=d.index)
        sig = sig.applymap(lambda x: pow(10, x))

        output = pd.concat([d, val, sig], axis=1)


        output['flag'] = output['next_val']- output['est']*5
        output['next_val'] = np.where(output['flag']>0,output['next_val'],output['est']*5)
        output['post_val'] = np.where(output['flag']>0,output['post_val'],output['est']*5)

        output['next_val_lower'] = output.eval('next_val / next_val_sig')
        output['next_val_upper'] = output.eval('next_val * next_val_sig')
        output['post_val_upper'] = output.eval('post_val * post_val_sig')

        df_v = output[['financing_date', 'next_val','next_val_lower',  'next_val_upper']]
        #
        #
        # df_v = pd.DataFrame(
        #     {'v_growth': output['post_val_upper'],
        #      'v_avg': output['next_val'],
        #      'v_min': output['next_val_lower'],
        #      'v_max': output['next_val_upper'],
        #      'financing_date':output['financing_date']})
        df_v['financing_date'] = df_v['financing_date'].apply(lambda x: x.strftime("%Y-%m-%d"))

        return df_v


if __name__ == '__main__':
    target_company = ['上海拉扎斯信息科技有限公司', '字节跳动有限公司', '紫光国芯微电子股份有限公司',
                      '北京石头世纪科技股份有限公司', '蚂蚁金服（杭州）网络技术有限公司', '宁德时代新能源科技股份有限公司',
                      '宁波容百新能源科技股份有限公司', '中简科技股份有限公司', '农夫山泉股份有限公司',
                      '稳健医疗用品股份有限公司', '江苏中信博新能源科技股份有限公司', '北京拜克洛克科技有限公司',
                      '名创优品（广州）有限责任公司', '舍得酒业股份有限公司', '北京畅行信息技术有限公司',
                      '上海鑫蓝地生物科技股份有限公司', '北京五八信息技术有限公司', '广州虎牙信息科技有限公司']
    check = ['上海夏实信息科技有限公司','上海携程商务有限公司','上海摄提信息科技有限公司',
             '上海艾力斯医药科技有限公司','上海赞荣医药科技有限公司','北京小马智行科技有限公司','深圳市前海第四范式数据技术有限公司',
             '深圳市耐能人工智能有限公司','深圳市萱嘉生物科技有限公司','苏州贝康医疗器械有限公司',
             '荣昌生物制药烟台股份有限公司','青岛创新奇智科技集团有限公司']
    raw = pd.DataFrame()
    vf = ValFinance(target_company = [],n_day = 0,date = 20210602)
    df = vf.get_data()
    print(df)


    vf.pred_many(df=df,if_last=1)
    # vf.pred_flask(data = df)








