"""
训练入口
分批生成模型训练集
训练模型

"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from calendar import monthrange
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
from utils import *
from tensorflow.keras.utils import plot_model
from LSTM import get_model
from tensorflow.keras.optimizers import Nadam
import os

def get_batch(n=100):
	flag = 'w'
	x_vals = list()
	x_labs = list()
	y_list = list()
	return_shift = np.random.normal(0, 0.05)  # 每个公司的特征回报率
	while n:
		# 成立公司
		year = np.random.randint(2000, 2021)
		month = np.random.randint(1, 13)
		day = np.random.randint(1, monthrange(year, month)[1] + 1)
		date = pd.Timestamp(year, month, day)
		# print('随机成立时间', date)
		# 初始股价
		price = 1.0
		share = round(np.random.lognormal(np.log(100), np.log(15)) + 2, 2)
		# print('随机注册资本/万元', price * share)
		# 融资轮次
		fin_round = 0
		# 每轮稀释比例下限; 上限; 每轮（到下一轮）的平均收益率
		round_specific = [('天使', 0.10, 0.25, 0.71),
						  ('种子', 0.10, 0.25, 0.12),
						  ('众筹', 0.10, 0.25, 0.45),
						  ('A', 0.01, 0.15, 2.37),
						  ('B', 0.01, 0.15, 1.31),
						  ('C', 0.01, 0.10, 1.04),
						  ('D', 0.01, 0.10, 0.72),
						  ('E', 0.01, 0.07, 0.92),
						  ('F', 0.01, 0.07, 0.336),
						  ('IPO', 0.01, 0.05, 0.15)]

		data = []
		today = datetime.now()
		# 统计平均融资间隔
		interval = int(np.random.exponential(1.09*365))
		round_sp = None

		while date < today:
			date += timedelta(days=interval)
			try:
				round_sp = round_specific[int(fin_round)]
			except:
				break
			offer_pct = np.random.uniform(round_sp[1], round_sp[2])
			offer = offer_pct * share  # 发行股份数
			share += offer
			# return_rate = np.random.normal(round_sp[3] + return_shift, 0.2)  # 期间回报率
			# period_return = np.max(pow(1 + return_rate, np.sqrt(interval / 365)),-1)
			# price = np.round(price * period_return, 2)

			miu = round_sp[3] + return_shift
			base = 0.06
			t = interval/365
			r = (2*miu - base)/(1+np.exp(t)) + base/(1+np.exp(t))
			price*=np.exp(np.random.normal((r- 0.2**2/2)*t, 0.2*t**0.5))
			post_val = price * share
			if post_val > 1e11:
				break
			fin_amount = offer * price
			fin_round += np.random.binomial(1, 0.6)  # 概率进入下一轮

			data.append([date, fin_amount, post_val, round_sp[0]])

			interval = int(np.random.exponential(1.09*365))

		if len(data) == 0:
			continue

		check_return_rate = np.random.normal(round_sp[3], 0.2)  # 检查时间点回报率
		check_period_return = max(pow(1 + check_return_rate, np.sqrt(interval / 365)),-1)
		check_price = round(price * check_period_return, 2)

		date += timedelta(days=interval)
		data.append([date, 1, check_price * share, 'check'])

		data = pd.DataFrame(data, columns=['date', 'amount', 'post_val', 'round'])
		data['about'] = data['amount'].apply(to_describe)

		data = data.groupby('round', as_index=False).apply(split_round)
		data = data.sort_values(by='date', ignore_index=True)
		data.to_csv('fake_data.csv', mode=flag, header=(flag=='w'), index=False)
		flag = 'a'

		x_interval = (data['date'].shift(-1) - data['date']).iloc[:-1].apply(lambda x: x.days / 365).rename(
			'x_interval')
		x_round = data['round'].apply(determine_round).rename('x_round')
		x_round = x_round.ffill()
		x_currency = data['about'].apply(determine_currency).rename('x_currency')
		x_val_mag = data['about'].apply(determine_value_mag)

		data = pd.concat([data, x_interval, x_round, x_currency, x_val_mag], axis=1)
		stop = data.query('x_round == -1')
		if len(stop):
			stop_loc = data.index.to_list().index(stop.index[0])
			data = data.iloc[:stop_loc]

		if 0 in data.iloc[-2:][['amount', 'post_val']].values:
			continue
		y_amount = np.log10(data.iloc[-2]['amount'])
		y_post_val = np.log10(data.iloc[-2]['post_val'])
		y_now_val = np.log10(data.iloc[-1]['post_val'])

		y_list.append([y_amount, y_post_val, y_now_val])
		x_vals.append(data.iloc[:-1][['x_interval', 'x_val', 'x_mag', 'x_currency']].values)
		x_labs.append(data.iloc[:-1][['x_round', 'x_expr']].values)
		n -= 1

	return pad_sequences(x_vals, 10, float), pad_sequences(x_labs, 10, int), np.array(y_list, float)
if __name__ =="__main__":
	model = get_model(Nadam(), loss=weighted_mse, metrics=[weighted_mae])

	try:
		model.load_weights('model.h5')
		flag = 'a'
		history_batches = pd.read_csv('history.csv').shape[0]
		print('Load history weights!')
	except:
		flag = 'w'
		history_batches = 0
		model.summary()
		plot_model(model, 'model.png', show_shapes=True, expand_nested=True, dpi=300)

	batch_size = 64
	batches = 2500
	res_list = list()
	min_loss = np.inf
	patient = 0
	for i in range(history_batches, batches + history_batches):
		X_val_train, X_lab_train, y_train = get_batch(batch_size)
		res = model.train_on_batch([X_val_train, X_lab_train], y_train)
		res_list.append(res)
		print('\rBatches: %d  loss = %1.4f  metrics = %1.4f' % (i + 1, *res), end=' ' * 10)

		if (i + 1) % 20 == 0:
			res_df = pd.DataFrame(res_list, columns=['loss', 'metrics'])
			res_list = list()
			res_mean = res_df.mean().values
			print('\rBatches: %d  loss = %1.4f  metrics = %1.4f' % (i + 1, *res_mean), end='  ')
			res_df.to_csv('history.csv', header=(flag=='w'), index=False, mode=flag)
			X_val_test, X_lab_test, y_test = get_batch(batch_size)

			test_loss = model.evaluate([X_val_test, X_lab_test], y_test, verbose=0)
			print('val_loss = %1.4f  val_metrics = %1.4f' % tuple(test_loss))
			res_df = pd.DataFrame([[i] + test_loss], columns=['batch', 'loss', 'metrics'])
			res_df.to_csv('test_history.csv', header=(flag=='w'), index=False, mode=flag)
			flag = 'a'

			plt.style.use('seaborn')
			fig, ax = plt.subplots(figsize=(12, 6))
			history = pd.read_csv('history.csv')
			ax.plot(history.index, history['loss'], label='Train Loss')
			ax.plot(history.index, history['metrics'], label='Train Metrics')

			history = pd.read_csv('test_history.csv')
			ax.plot(history['batch'], history['loss'], label='Test Loss')
			ax.plot(history['batch'], history['metrics'], label='Test Metrics')

			ax.legend()
			ax.set_title('Loss History')
			ax.set_xlabel('Batches')
			ax.set_ylim(0)
			ax.set_xlim(0)

			plt.savefig('history.png', format='PNG')
			plt.close()

			loss_dec = min_loss - test_loss[0]
			min_loss = min(min_loss, test_loss[0])
			if (i + 1) % 200 == 0:
				print('Save Model at %d batch' % (i + 1))
				model.save(os.path.join(os.getcwd(), 'model', 'lstm.h5'))

	model.save(os.path.join(os.getcwd(), 'model', 'lstm.h5'))
