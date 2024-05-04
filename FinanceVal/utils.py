import pymongo
import re
import numpy as np
import pandas as pd
import re
import numpy as np
import pandas as pd
import tensorflow.keras.backend as K
from pymongo import MongoClient
from CONFIG.Config import config
import time
import datetime

__all__ = ['to_about', 'to_exact', 'to_describe', 'determine_round',
		   'determine_value_mag', 'determine_currency',
           'split_round', 'weighted_mae', 'weighted_mse']


def approx(n):
	if n ==0:
		mag = 0
	else:
		mag = np.log10(n)
	if 0.3 < mag - int(mag) < 0.95:
		if np.random.binomial(1, 0.4):
			return '数' + ['万', '十万', '百万', '千万', '亿', '十亿', '百亿', '千亿', '万亿'][min(int(mag),8)]

	dn = n / pow(10, int(mag))
	rn = round(dn, np.random.randint(2))
	pn = rn * pow(10, int(mag))
	if rn > dn:
		if mag > 4:
			pn /= 1e4
			if pn > 10:
				return '近%d亿' % pn
			else:
				val = '%.1f' % pn
				if val.endswith('.0'):
					val = val[:-2]
				return '近%s亿' % val
		else:
			if pn > 10:
				return '近%d万' % pn
			else:
				val = '%.1f' % pn
				if val.endswith('.0'):
					val = val[:-2]
				return '近%s万' % val
	else:
		word = '超逾过'[np.random.randint(3)]
		if mag > 4:
			pn /= 1e4
			if pn > 10:
				return word + '%d亿' % pn
			else:
				val = '%.1f' % pn
				if val.endswith('.0'):
					val = val[:-2]
				return word + '%s亿' % val
		else:
			if pn > 10:
				return word + '%d万' % pn
			else:
				val = '%.1f' % pn
				if val.endswith('.0'):
					val = val[:-2]
				return word + '%s万' % val


def to_describe(x):
	if np.random.binomial(1, 0.4):
		x /= 6.4
		unit = '美元'
	else:
		unit = '元人民币'
	rand = np.random.rand()
	mag = np.log10(x)
	if rand < 0.27:
		return ['未披露', '未公开'][np.random.randint(2)]
	elif rand < 0.71:
		return approx(x) + unit
	else:
		digit = np.random.randint(3)
		if mag > 4:
			cx = x / 1e4
			return ('%%.%df亿' % digit) % cx + unit
		else:
			return ('%%.%df万' % digit) % x + unit


# val = np.power(10, np.random.rand(100) * 6)
# list(map(to_describe, val))

def to_about(x):
	mag = np.log10(x)
	im_dict = ['万', '十万', '百万', '千万', '亿', '十亿', '百亿', '千亿']
	im, fm = int(mag), mag - int(mag)
	if fm > 0.94:
		about = '近'
		im += 1
	elif fm < 0.1:
		about = ['超', '逾', '过', '级'][np.random.randint(0, 4)]
	else:
		about = '数'

	if about != '级':
		return about + im_dict[im]
	else:
		return im_dict[im] + about


def to_exact(x):
	mag = np.log10(x) / 4
	im = int(mag)
	im_dict = ['万', '亿']
	ix = round(x / pow(10, im * 4), np.random.randint(0, 2)) * pow(10, im * 4)
	about = ('%.2f' % (ix / pow(10, im * 4)))[:4]
	while about.endswith(('.000', '.00', '.0', '.')):
		about = about[:-1]
	return about + im_dict[im]

# 估值 将真实轮次转为数字
# 特殊轮次赋予特殊值

def determine_round(x):
	try:
		find = re.findall(r'(pre)?[-]?(天使|种子|众筹|IPO|战略投资|私有化)?([A-Z])?([+3-9]*)?', x)[0]
		if find[0] == '' and find[1] == 'IPO':
			return np.nan  # IPO轮不反馈
		elif find[1] == 'IPO':
			return 10
		elif find[1] in ['众筹', '天使', '种子']:
			return 1
		elif find[1] == '战略投资':
			return np.nan
		elif find[1] == '私有化':
			return 100
		else:
			return min(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ').index(find[2]) + 2, 9)
	except:
		return 0

# 估值 识别真实数据中的汇率
def determine_currency(x):
	try:
		find = re.findall(r'(美元)', x)[0]
		return 1
	except:
		return 0

# 估值 转换真实数据中的 数字规模 数值 前缀 并提取有效真实数值
def determine_value_mag_real(x):
	index = ['x_val', 'x_expr', 'x_mag', 'est']
	try:
		find = re.findall(r'(美元)', x)[0]
		ex = 6.7
	except:
		ex = 1
	try:
		find = re.findall(r'([0-9.,]*)?([数超过逾近])?([十百千万亿]*)', x)[0]
		val, expr, mag = find
		val = find[0].replace(',', '')
		expr = ['', '近', '超', '过', '逾', '数'].index(expr)
		unit = ['', '万', '十万', '百万', '千万', '亿', '十亿', '百亿', '千亿', '万亿'].index(mag)

		try:
			val = float(val)
		except:
			val = 0

		while val >= 10:
			val /= 10
			unit += 1
		while 0 < val < 1:
			val *= 10
			unit -= 1
		if expr == 5:
			val = 3
		elif expr == 1:
			val = 0.9 * val
		elif expr == 0:
			pass
		else:
			val = 1.1 * val

		if unit >= 1:
			est = val * 10000 * 10 ** (unit - 1) * ex
		else:
			est = val * 10000 * ex

		return pd.Series([val, expr, unit, est], index=index)
	except:
		return pd.Series([0, 0, 0, 0], index=index)

# 估值 真实数据转为 输入项
def to_data(df, until=None):
	reject = ['并购', '上市公司定增', '股权转让', '债权融资', '定向增发']
	until = df.query('"IPO" in turn')['financing_date']
	if not until.empty:
		until = until.values[0]
		df = df.query('financing_date <= @until')
	df = df.query('turn not in @reject')

	df['turn'] = df['turn'].apply(lambda x: x.replace('新三板定增', 'F轮'))

	df['interval'] = (df['financing_date'].shift(-1).fillna(datetime.datetime.now()) - df['financing_date']).apply(
		lambda x: x.days / 365)
	df = df.query('turn != "IPO"')
	df['round'] = df['turn'].apply(determine_round)
	df['round'] = df['round'].ffill()
	df['currency'] = df['announce'].apply(determine_currency)
	df[['val', 'expr', 'mag', 'est']] = df['announce'].apply(determine_value_mag_real)
	"""
    你把有融资金额的数据金额换算出来（人民币万元），用融资估值除以金额，要求99%的数据在倍数在3-100倍以内。
    没有具体金额的，用4代表数，0.9代表近，1.5代表超
    """
	df = df.reset_index(drop=True)
	if not df.empty:
		for i in df.columns:
			try:
				df[i] = df[i].fillna(0)
			except:
				df[i] = df[i].fillna(pd.Timedelta(seconds=0))
		return df
	else:
		return pd.DataFrame()



# 假数据
def split_round(df):
	if len(df) == 1:
		df['round'] = df['round'].apply(lambda x: x + '轮')
		return df
	else:
		new_df = list()
		rand = np.random.randint(0, 2)
		for i in range(df.shape[0]):
			new_ser = df.iloc[i]
			if i < rand:
				new_ser['round'] = 'pre-' + new_ser['round']
			elif i - rand == 1:
				new_ser['round'] = new_ser['round'] + '+'
			elif i - rand == 2:
				new_ser['round'] = new_ser['round'] + '++'
			elif i > rand:
				new_ser['round'] = new_ser['round'] + format(i - rand, 'd')
			new_df.append(new_ser)
		new_df = pd.DataFrame(new_df)
		new_df['round'] = new_df['round'].apply(lambda x: x + '轮')
		return new_df
# 假数据
def determine_round(x):
	try:
		find = re.findall(r'(pre)?[-]?(天使|种子|众筹|IPO|战略投资)?([A-Z])?([+3-9]*)?', x)[0]
		if find[0] == '' and find[1] == 'IPO':
			return -1  # IPO轮不反馈
		elif find[1] == 'IPO':
			return 10
		elif find[1] in ['众筹', '天使', '种子']:
			return 1
		elif find[1] == '战略投资':
			return np.nan
		else:
			return min(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ').index(find[2]) + 2, 9)
	except:
		return 0

# 假数据
def determine_value_mag(x):

	index = ['x_val', 'x_expr', 'x_mag']
	try:
		find = re.findall(r'([数超过逾近约])?([0-9.,]*)?([十百千万亿]*)', x)[0]
		expr,val, mag = find
		lmag = [i for i in mag]
		mag = list(set(lmag))
		mag.sort(key=lmag.index)
		mag = ''.join(mag)

		val = find[1].replace(',', '')
		expr = ['', '近', '超', '过', '逾', '数','约'].index(expr)
		unit = ['', '万', '十万', '百万', '千万', '亿', '十亿', '百亿', '千亿', '万亿'].index(mag)

		try:
			val = float(val)
		except:
			val = 0

		while val >= 10:
			val /= 10
			unit += 1
		while 0 < val < 1:
			val *= 10
			unit -= 1


		return pd.Series([val, expr, unit], index=index)
	except Exception as e:
		print(e)
		return pd.Series([0, 0, 0], index=index)
# 模型
def weighted_mae(y_true, y_pred):
	y_val_pred = y_pred[:, 0, :]
	y_sig_pred = y_pred[:, 1, :]
	d_val = K.abs(y_true - y_val_pred)
	d_sig = K.abs(d_val - y_sig_pred)
	wd_val = K.mean(d_val, axis=0) * K.constant([0.4, 0.4, 0.2])
	wd_sig = K.mean(d_sig, axis=0) * K.constant([0.4, 0.4, 0.2])
	return wd_val * 0.8 + wd_sig * 0.2
# 模型
def weighted_mse(y_true, y_pred):
	y_val_pred = y_pred[:, 0, :]
	y_sig_pred = y_pred[:, 1, :]
	d_val = K.square(y_true - y_val_pred)
	d_sig = K.square(K.sqrt(d_val) - y_sig_pred)
	wd_val = K.mean(d_val, axis=0) * K.constant([0.4, 0.4, 0.2])
	wd_sig = K.mean(d_sig, axis=0) * K.constant([0.4, 0.4, 0.2])
	return wd_val * 0.8 + wd_sig * 0.2


# 读取mongo
def read_mongo(db_name, collection_name, conditions={}, query=[], sort_item='', n=0):
	mongo_uri_test = 'mongodb://%s:%s@%s:%s/?authSource=%s' % \
					 (config['DEFAULT']['mongo_user'],
					  config['DEFAULT']['mongo_password'],
					  config['DEFAULT']['mongo_host'],
					  int(config['DEFAULT']['mongo_port']), db_name)
	conn = MongoClient(mongo_uri_test)

	db = conn[db_name]
	collection = db[collection_name]
	if query:
		cursor = collection.find(conditions, query)
	else:
		cursor = collection.find(conditions)
	if sort_item:
		cursor = cursor.sort(sort_item, pymongo.DESCENDING)
	if n:
		cursor = cursor[:n]
	temp_list = []
	for single_item in cursor:
		temp_list.append(single_item)

	df = pd.DataFrame(temp_list)
	conn.close()
	return df

# 存储mongo
def save_mongo(df_data, db_name, tb_name, if_del=0, del_condition={}):
	mongo_uri_test = 'mongodb://%s:%s@%s:%s/?authSource=%s' % \
					 (config['DEFAULT']['mongo_user'],
					  config['DEFAULT']['mongo_password'],
					  config['DEFAULT']['mongo_host'],
					  int(config['DEFAULT']['mongo_port']),db_name)
	conn = MongoClient(mongo_uri_test)
	db = conn[db_name]
	collection = db[tb_name]
	if if_del:
		if del_condition:
			collection.delete_many(del_condition)
		else:
			collection.delete_many({})
	list_data = df_data.to_dict('record')
	collection.insert_many(list_data)

# 获取当天日期
def get_today():
	today = time.mktime(datetime.datetime.now().date().timetuple())
	today = datetime.datetime.fromtimestamp(today)
	return today

# 获取n天前日期
def get_n_day_before(n=0):
	t = (datetime.datetime.now() - datetime.timedelta(days=n))
	start = t.year * 10000 + t.month * 100 + t.day
	return start