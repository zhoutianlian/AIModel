import numpy as np
import os
gics4 = np.array(
        [10101010, 10101020, 10102010, 10102020, 10102030, 10102040, 10102050, 15101010, 15101020, 15101030,
         15101040, 15101050, 15102010, 15103010, 15103020, 15104010, 15104020, 15104025, 15104030, 15104040,
         15104045, 15104050, 15105010, 15105020, 20101010, 20102010, 20103010, 20104010, 20104020, 20105010,
         20106010, 20106015, 20106020, 20107010, 20201010, 20201050, 20201060, 20201070, 20201080, 20202010,
         20202020, 20301010, 20302010, 20303010, 20304010, 20304020, 20305010, 20305020, 20305030, 25101010,
         25101020, 25102010, 25102020, 25201010, 25201020, 25201030, 25201040, 25201050, 25202010, 25203010,
         25203020, 25203030, 25301010, 25301020, 25301030, 25301040, 25302010, 25302020, 25501010, 25502020,
         25503010, 25503020, 25504010, 25504020, 25504030, 25504040, 25504050, 25504060, 30101010, 30101020,
         30101030, 30101040, 30201010, 30201020, 30201030, 30202010, 30202030, 30203010, 30301010, 30302010,
         35101010, 35101020, 35102010, 35102015, 35102020, 35102030, 35103010, 35201010, 35202010, 35203010,
         40101010, 40101015, 40102010, 40201020, 40201030, 40201040, 40202010, 40203010, 40203020, 40203030,
         40203040, 40204010, 40301010, 40301020, 40301030, 40301040, 40301050, 45102010, 45102020, 45102030,
         45103010, 45103020, 45201020, 45202030, 45203010, 45203015, 45203020, 45203030, 45301010, 45301020,
         50101010, 50101020, 50102010, 50201010, 50201020, 50201030, 50201040, 50202010, 50202020, 50203010,
         55101010, 55102010, 55103010, 55104010, 55105010, 55105020, 60101010, 60101020, 60101030, 60101040,
         60101050, 60101060, 60101070, 60101080, 60102010, 60102020, 60102030, 60102040]
)

gics_dict = {10101010: '石油与天然气钻井',
             10101020: '石油天然气设备与服务',
             10102010: '综合性石油与天然气企业',
             10102020: '石油与天然气的勘探与生产',
             10102030: '石油与天然气的炼制和营销',
             10102040: '石油与天然气的储存和运输',
             10102050: '煤与消费用燃料',
             15101010: '商品化工',
             15101020: '多种化学制品',
             15101030: '化肥与农用药剂',
             15101040: '工业气体',
             15101050: '特种化学制品',
             15102010: '建筑材料',
             15103010: '金属与玻璃容器',
             15103020: '纸材料包装',
             15104010: '铝',
             15104020: '多种金属与采矿',
             15104025: '铜',
             15104030: '黄金',
             15104040: '贵重金属与矿石',
             15104045: '白银',
             15104050: '钢铁',
             15105010: '林业产品',
             15105020: '纸制品',
             20101010: '航天航空与国防',
             20102010: '建筑产品',
             20103010: '建筑与工程',
             20104010: '电气部件与设备',
             20104020: '重型电气设备',
             20105010: '工业集团企业',
             20106010: '建筑机械与重型卡车',
             20106015: '农用农业机械',
             20106020: '工业机械',
             20107010: '贸易公司与经销商',
             20201010: '商业印刷',
             20201050: '环境与设施服务',
             20201060: '办公服务与用品',
             20201070: '综合支持服务',
             20201080: '安全和报警服务',
             20202010: '人力资源与就业服务',
             20202020: '调查和咨询服务',
             20301010: '航空货运与物流',
             20302010: '航空公司',
             20303010: '海运',
             20304010: '铁路',
             20304020: '陆运',
             20305010: '机场服务',
             20305020: '公路与铁路',
             20305030: '海港与服务',
             25101010: '机动车零配件与设备',
             25101020: '轮胎与橡胶',
             25102010: '汽车制造商',
             25102020: '摩托车制造商',
             25201010: '消费电子产品',
             25201020: '家庭装饰品',
             25201030: '住宅建筑',
             25201040: '家用电器',
             25201050: '家用器具与特殊消费品',
             25202010: '消闲用品',
             25203010: '服装、服饰与奢侈品',
             25203020: '鞋类',
             25203030: '纺织品',
             25301010: '赌场与赌博',
             25301020: '酒店、度假村与豪华游轮',
             25301030: '消闲设施',
             25301040: '餐馆',
             25302010: '教育服务',
             25302020: '特殊消费者服务',
             25401010: '广告（自2018年9月28日休市后终止生效）',
             25401020: '广播（自2018年9月28日休市后起终止生效）',
             25401025: '有线和卫星电视（自2018年9月28日休市后终止生效）',
             25401030: '电影与娱乐（自2018年9月28日休市后终止生效）',
             25401040: '出版（自2018年9月28日休市后终止生效）',
             25501010: '经销商',
             25502020: '互联网与直销零售',
             25503010: '百货商店',
             25503020: '综合货品商店',
             25504010: '服装零售',
             25504020: '电脑与电子产品零售',
             25504030: '家庭装潢零售',
             25504040: '专卖店',
             25504050: '汽车零售',
             25504060: '家庭装饰零售',
             30101010: '药品零售',
             30101020: '食品分销商',
             30101030: '食品零售',
             30101040: '大卖场与超市',
             30201010: '啤酒酿造商',
             30201020: '酿酒商与葡萄酒商',
             30201030: '软饮料',
             30202010: '农产品',
             30202030: '包装食品与肉类',
             30203010: '烟草',
             30301010: '居家用品',
             30302010: '个人用品',
             35101010: '医疗保健设备',
             35101020: '医疗保健用品',
             35102010: '保健护理产品经销商',
             35102015: '保健护理服务',
             35102020: '保健护理机构',
             35102030: '管理型保健护理',
             35103010: '医疗保健技术',
             35201010: '生物科技',
             35202010: '制药',
             35203010: '生命科学工具和服务',
             40101010: '综合性银行',
             40101015: '区域性银行',
             40102010: '互助储蓄与抵押信贷金融服务',
             40201020: '其它综合性金融服务',
             40201030: '多领域控股',
             40201040: '特殊金融服务',
             40202010: '消费信贷',
             40203010: '资产管理与托管银行',
             40203020: '投资银行业与经纪业',
             40203030: '综合性资本市场',
             40203040: '金融交易所和数据',
             40204010: '抵押房地产投资信托',
             40301010: '保险经纪商',
             40301020: '人寿与健康保险',
             40301030: '多元化保险',
             40301040: '财产与意外伤害保险',
             40301050: '再保险',
             45101010: '互联网软件与服务（自2018年9月28日休市后终止生效）',
             45102010: '信息科技咨询与其它服务',
             45102020: '数据处理与外包服务',
             45102030: '互联网服务与基础架构',
             45103010: '应用软件',
             45103020: '系统软件',
             45103030: '家庭娱乐软件（自2018年9月28日休市后终止生效）',
             45201020: '通信设备',
             45202030: '电脑硬件、储存设备及电脑周边',
             45203010: '电子设备和仪器',
             45203015: '电子元件',
             45203020: '电子制造服务',
             45203030: '技术产品经销商',
             45301010: '半导体设备',
             45301020: '半导体产品',
             50101010: '非传统电信运营商',
             50101020: '综合电信业务',
             50102010: '无线电信业务',
             50201010: '广告',
             50201020: '广播',
             50201030: '有线和卫星',
             50201040: '出版',
             50202010: '电影和娱乐',
             50202020: '互动家庭娱乐',
             50203010: '互动媒体与服务',
             55101010: '电力公用事业',
             55102010: '燃气公用事业',
             55103010: '复合型公用事业',
             55104010: '水公用事业',
             55105010: '独立电力生产商与能源贸易商',
             55105020: '新能源发电业者',
             60101010: '多样化房地产投资信托',
             60101020: '工业房地产投资信托',
             60101030: '酒店及度假村房地产投资信托 ',
             60101040: '办公房地产投资信托',
             60101050: '医疗保健房地产投资信托',
             60101060: '住宅房地产投资信托',
             60101070: '零售业房地产投资信托',
             60101080: '特种房地产投资信托',
             60102010: '多样化房地产活动',
             60102020: '房地产经营公司',
             60102030: '房地产开发',
             60102040: '房地产服务'}

def concat_gics(sub_list):
    sup_list = list()
    for g in sub_list:
        if len(sup_list) == 0:
            sup_list.append(g//100)
        elif sup_list[-1] != g//100:
            sup_list.append(g//100)
    return sup_list

gics3 = np.array(concat_gics(gics4))
gics2 = np.array(concat_gics(gics3))
gics1 = np.array(concat_gics(gics2))

y3 = list()
for i in gics3:
    index = np.where(gics4 // 100 == i)[0]
    start, count = index[0], len(index)
    y3.append(slice(start, start + count))

y2 = list()
for i in gics2:
    index = np.where(gics4 // 10000 == i)[0]
    start, count = index[0], len(index)
    y2.append(slice(start, start + count))

y1 = list()
for i in gics1:
    index = np.where(gics4 // 1000000 == i)[0]
    start, count = index[0], len(index)
    y1.append(slice(start, start + count))

BERT_CONFIG_PATH = r'D:\python_zhoutianlian\tfnnStudy\bert\publish\bert_config.json'
BERT_CKPT_PATH = r'D:\python_zhoutianlian\tfnnStudy\bert\publish\bert_model.ckpt'
VOCAB_PATH = r'D:\python_zhoutianlian\tfnnStudy\bert\publish\vocab.txt'