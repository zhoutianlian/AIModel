import pandas as pd
from datetime import datetime
from utils import determine_round, determine_currency, determine_value_mag, weighted_mae, weighted_mse
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from LSTM import get_model
from valuation_by_financing import ValFinance


def to_data(df, until=None):
	until = until or datetime.now()
	reject = ['并购', '上市公司定增', '股权转让']
	df = df.query('financing_date <= @until ')
	df = df.query('turn not in @reject')
	df['interval'] = (df['financing_date'].shift(-1).fillna(datetime.now()) - df['financing_date']).apply(lambda x: x.days / 365)
	df = df.query('turn != "IPO"')
	df['round'] = df['turn'].apply(determine_round)
	df['round'] = df['round'].ffill()
	df['currency'] = df['announce'].apply(determine_currency)
	df[['val', 'expr', 'mag']] = df['announce'].apply(determine_value_mag)
	df = df.reset_index(drop=True)
	if not df.empty:
		for i in df.columns:
			print(df[i])
			try:
				df[i] = df[i].fillna(0)
			except:
				df[i] = df[i].fillna(pd.Timedelta(seconds=0))
		return df
	else:
		return pd.DataFrame()

def evl_lstm(data,model,if_add = 1):
	output = list()
	if if_add:
		flag = 'a'
	else:
		flag = 'w'
	for i in data.index.levels[0]:
		try:
			d = data.loc[i]
			x_val = list()
			x_lab = list()
			index = list()
			for j in range(len(d)):
				x_val.append(d.iloc[:j+1][['interval', 'val', 'mag']])
				x_lab.append(d.iloc[:j+1][['round', 'currency', 'expr']])
				index.append(d.iloc[j]['financing_date'])
			x_val = pad_sequences(x_val, 10, float)
			x_lab = pad_sequences(x_lab, 10, float)
			y = model.predict([x_val, x_lab])
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
			output.to_csv('./output/result.csv', header=(flag=='w'), mode=flag, index=False, encoding='gbk')
			flag = 'a'
		except Exception as e:
			print(i, e)
			continue
if __name__ == '__main__':
	# data = pd.read_csv('data_orig.csv', index_col=0)
	# data['financing_date'] = data['financing_date'].apply(pd.Timestamp)
	# data = data.sort_values(by=['company_name', 'financing_date'], ignore_index=True)
	# data = data.groupby('company_name', as_index=True).apply(to_data)


	target_company = ['上海拉扎斯信息科技有限公司', '字节跳动有限公司', '紫光国芯微电子股份有限公司',
					  '北京石头世纪科技股份有限公司', '蚂蚁金服（杭州）网络技术有限公司', '宁德时代新能源科技股份有限公司',
					  '宁波容百新能源科技股份有限公司', '中简科技股份有限公司', '农夫山泉股份有限公司',
					  '稳健医疗用品股份有限公司', '江苏中信博新能源科技股份有限公司', '北京拜克洛克科技有限公司',
					  '名创优品（广州）有限责任公司', '舍得酒业股份有限公司', '北京畅行信息技术有限公司',
					  '上海鑫蓝地生物科技股份有限公司', '北京五八信息技术有限公司', '广州虎牙信息科技有限公司']

	data = pd.DataFrame()
	vf = ValFinance(target_company=[], n_day=30)
	df = vf.get_data()
	for i, x in df.iterrows():
		print(x['company_name'])
		name = x['company_name']
		x = vf.clean_data(x)
		x['company_name'] = name
		data = data.append(x)
	data['financing_date'] = data['financing_date'].apply(pd.Timestamp)
	data = data.sort_values(by=['company_name', 'financing_date'], ignore_index=True)
	data = data.groupby('company_name', as_index=True).apply(to_data)

	model = load_model('./model/lstm.h5',
					   custom_objects={'weighted_mse': weighted_mse, 'weighted_mae': weighted_mae})

	evl_lstm(data=data, model=model,if_add=1)

"""
{
    "financing_date":[20190331,20170901,20160408,20160204,20150907,20150304],
    "financing_amount":[6.588900e+06,2.206038e+07,1.627163e+08,5.258540e+07,4.636000e+04,1.090000e+05 ],
    "turn":["定向增发","股权转让","B轮","股权转让","股权转让","定向增发"],
    "announce":["658.89万人民币","341.665万美元","1.627亿人民币","5258.54万元","4.636万人民币","10.9万人民币"]
}
"""