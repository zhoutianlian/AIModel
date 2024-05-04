from pymongo import MongoClient
import pymysql
import time
from pandas import Timestamp

from ValuationRecord import SaveEnterprise, GetTableId, GetValuationPeer
from CONFIG.globalENV import global_var


def start(enter):
    user_id = 1
    my_db = pymysql.connect(global_var.name, global_var.user, global_var.password, 'rdt_fintech')
    cursor = my_db.cursor()
    vid = 0
    try:
        # 执行存储企业操作,获取enterpriseId
        enterprise_id = SaveEnterprise.save(enter['enterprise_name'])
        if enterprise_id is not None:
            valuation_input = {}
            # 估值类型
            val_type = int(enter['type'])
            # 获取融资轮次，所在市场，社保人数，资质
            basic = eval(enter['basic'])
            valuation_input['inputRound'] = basic['round']
            valuation_input['inputMarket'] = basic['market']
            valuation_input['inputStaffNum'] = basic['staff']
            valuation_input['inputQual'] = basic['qualify']
            # 获取行业和对标公司{"10101010":[0.7,[10,12,14,15,35,38,34]],……}
            ind_codes = eval(enter['industry'])
            industry = {}
            # 对标公司
            peer_ids = GetValuationPeer.get_peer(eval(enter['peer']))
            # industry_percents = []
            for code in ind_codes:
                industry[str(code)] = [float(ind_codes[code]), peer_ids[code]]

            valuation_input['inputIndustry'] = industry
            # 估值模式
            valuation_input['inputValMode'] = int(enter['mode'])
            conn = MongoClient('mongodb://hub:hubhub@'+global_var.name+':%s/' % global_var.port)
            db = conn.rdt_fintech
            # 风投模型（暂无）
            valuation_input['vcId'] = 0
            # 商业模型
            bm = eval(enter['bm'])
            if len(bm) != 0:
                table_bm = db.NewBm
                bm_num = GetTableId.get_table_id('NewBm')
                bm['_id'] = bm_num
                table_bm.insert(bm)
                valuation_input['bmId'] = bm_num
            else:
                valuation_input['bmId'] =0
            # 预测
            table_hypo = db.Hypo
            hypo_num = GetTableId.get_table_id('Hypo')
            hypo = eval(enter['hypo'])
            hypo['_id'] = hypo_num
            table_hypo.insert(hypo)
            valuation_input['hypoId'] = hypo_num
            # 财务数据id集合
            bal_ids = []
            table_bal = db.NewBal
            bal_df = eval(enter['bal'])
            for bal_data in bal_df:
                bal_num = GetTableId.get_table_id('NewBal')
                bal_data['_id'] = bal_num
                table_bal.insert(bal_data)
                bal_ids.append(bal_data['_id'])
            valuation_input['balId'] = bal_ids
            flow_ids = []
            table_flow = db.NewFlow
            flow_df = eval(enter['flow'])
            for flow_data in flow_df:
                flow_num = GetTableId.get_table_id('NewFlow')
                flow_data['_id'] = flow_num
                table_flow.insert(flow_data)
                flow_ids.append(flow_data['_id'])
            valuation_input['flowId'] = flow_ids
            # 活动id，货币id，公司id，用户id，估值精度，输入方式，IP地址，终端，估值类型，估值有效性
            sql = "INSERT INTO t_valuating_record (activity_id, currency_id, enterprise_id, user_id, val_accuracy, " \
                  "val_inputmethod, val_ip, val_terminal, val_type, val_valid, channel_id, c_time) VALUES " \
                  "(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
            now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            val = (0, 1, str(enterprise_id), str(user_id), 0, 12, "116.233.236.80", 28, val_type, 0, 0, now)
            try:
                cursor.execute(sql, val)
                my_db.commit()
                vid = cursor.lastrowid
                valuation_input['_id'] = vid
                table_input = db.ValuatingInput
                table_input.insert(valuation_input)
            except KeyboardInterrupt:
                my_db.rollback()
            finally:
                my_db.close()
    except Exception as e:
        print(e)
    return vid
