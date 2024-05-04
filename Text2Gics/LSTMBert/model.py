"""
模型实现
训练功能
预测功能

"""
from settings import *
import os

if not USE_GPU:
	os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
	import tensorflow as tf
else:
	os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
	import tensorflow as tf
	gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
	for gpu in gpus:
		tf.config.experimental.set_memory_growth(gpu, True)

import keras_bert
import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from config import BERT_CONFIG_PATH, BERT_CKPT_PATH, gics4
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.regularizers import l1
from review import review
from data import data_generator
from data import TEXT_COL, pre_treat, join, get_token
from tensorflow.keras.preprocessing.sequence import pad_sequences
from settings import MAX_LENGTH

pd.options.display.float_format = '{:.4f}'.format


class IndustryClassifierModel:
	def __init__(self, structure, activation='relu', detail=True):
		"""
		创建模型
		:param structure: 自定义模型部分
		:param activation: 自定义模型部分激活函数
		"""
		self._gics4_count = len(gics4)
		self._bert_shape = None

		# 读取预训练的bert
		self._bert = keras_bert.load_trained_model_from_checkpoint(BERT_CONFIG_PATH, BERT_CKPT_PATH, trainable=False)
		self._build_model(structure, activation)  # 创建模型

		if detail:
			self._sub_model.summary()
			self._full_model.summary()

	def  _log_model(self, log_path):
		pass

	def _get_input(self, _id: int=0):
		input_1 = Input(shape=(BERT_LEN,), name='Input%d1' % _id)
		input_2 = Input(shape=(BERT_LEN,), name='Input%d2' % _id)
		vec = self._bert([input_1, input_2])
		return [input_1, input_2], vec

	def _build_model(self, structure, activation):

		tok_input = Input(shape=(BERT_LEN,), name='TOK_INPUT')
		seg_input = Input(shape=(BERT_LEN,), name='SEG_INPUT')

		bert_output = self._bert([tok_input, seg_input])
		shape = bert_output.shape[1:]

		sub_input, sub_output = structure(shape, activation)

		output = Dense(self._gics4_count,
			           activity_regularizer=l1(),
			           use_bias=False,
			           name='Dense3')(sub_output)

		self._sub_model = Model(sub_input, output, name='SUB_MODEL')
		self._full_model = Model([tok_input, seg_input], self._sub_model(bert_output), name='FULL_MODEL')

	def train(self, save_path: str, loss,
	          metrics=None,
	          optimizer=Nadam(),
	          batches: int = 50000,
	          batch_size: int = 32,
	          start: int = 0,
	          valid_batches: int = 10,
	          save_period: int = 500,
	          eval_period: int = 50,
	          exchange_frequency: int = 0,
	          ):

		self._full_model.compile(optimizer, loss, metrics)

		train_data = data_generator(TRAIN_PATH, exchange_frequency=exchange_frequency)
		next(train_data)
		valid_data = data_generator(VALID_PATH, exchange_frequency=exchange_frequency)
		next(valid_data)

		train_log = list()
		epoch_log = list()

		log_mode = 'w'
		x_test, y_test = train_data.send(batch_size)
		print('Shape: tok = %s  seg = %s  y = %s' % (x_test[0].shape, x_test[1].shape, y_test.shape))

		save_path = 'save/' + save_path
		if not os.path.exists(save_path):
			os.makedirs(save_path + '/weights')

		if start > 0:
			try:
				self._sub_model.load_weights(save_path + '/weights/model%08d.h5' % start)
				print('Load weights from %08d model!' % start)

				# 重置历史记录
				train_history = pd.read_csv(save_path + '/train_history.csv', index_col=0)
				train_index = np.array(train_history.index)
				train_history.loc[train_index[train_index <= start]].to_csv(save_path + '/train_history.csv')
				print('Reset train history to batch %d!' % start)

				valid_history = pd.read_csv(save_path + '/valid_history.csv', index_col=0)
				valid_index = np.array(valid_history.index)
				valid_history.loc[valid_index[valid_index <= start]].to_csv(save_path + '/valid_history.csv')
				print('Reset valid history to batch %d!' % start)

				# 重置图片
				review(save_path, eval_period)
				print('Reset review plots to batch %d!' % start)

				# 修改记录模式
				log_mode = 'a'
			except Exception as err:
				print(err)
				raise FileNotFoundError('Cannot find model %08d' % start)
		else:
			self._log_model(save_path)

		mark_batch = start

		def save_model(now_batch):
			self._sub_model.save_weights(save_path + '/weights/model%08d.h5' % now_batch)
			print('========== Save model %08d! ==========' % now_batch)
			x_, y_ = train_data.send(batch_size * 100)
			train_eval = self._full_model.evaluate(x_, y_)
			train_eval = pd.Series(train_eval, index=['loss', 'acc1', 'acc2', 'acc3', 'acc4'])
			x_, y_ = valid_data.send(batch_size * 100)
			valid_eval = self._full_model.evaluate(x_, y_)
			valid_eval = pd.Series(valid_eval, index=['loss', 'acc1', 'acc2', 'acc3', 'acc4'])
			eval_result = pd.DataFrame([train_eval, valid_eval], index=['train', 'valid'])

			print(eval_result)
			print('*' * 90)

		save = True
		try:
			for b in range(start, start + batches):
				mark_batch = b + 1
				save = True
				x, y = train_data.send(batch_size)
				result = self._full_model.train_on_batch(x, y)
				train_log.append(result)
				epoch_log.append(mark_batch)
				print('\rBatch %08d  ' % mark_batch, end='')
				print('loss = %.4f  acc1 = %.4f  acc2 = %.4f  acc3 = %.4f  acc4 = %.4f  ' % tuple(result), end='')

				if mark_batch % eval_period == 0:
					print('\n--- Batch %08d to %08d Summary ---' % (epoch_log[0], epoch_log[-1]))
					train_df = pd.DataFrame(train_log, columns=['loss', 'acc1', 'acc2', 'acc3', 'acc4'], index=epoch_log)
					train_df.to_csv(save_path + '/train_history.csv', header=(log_mode == 'w'), mode=log_mode)
					train_summary = train_df.mean()

					x, y = valid_data.send(valid_batches*batch_size)
					eva_result = self._full_model.evaluate(x, y, verbose=0)
					valid_summary = pd.Series(eva_result, index=['loss', 'acc1', 'acc2', 'acc3', 'acc4'])
					valid_summary = valid_summary.rename(mark_batch)
					valid_df = valid_summary.to_frame().T
					valid_df.to_csv(save_path + '/valid_history.csv', header=(log_mode=='w'), mode=log_mode)

					summary_df = pd.DataFrame([train_summary, valid_summary], index=['train', 'valid'])
					print(summary_df)
					print('*' * 90)

					train_log = list()
					epoch_log = list()
					log_mode = 'a'

					review(save_path, eval_period)

				if mark_batch % save_period == 0:
					save_model(mark_batch)
					save = False
		except Exception as err:
			print(err)
		finally:
			if save:
				save_model(mark_batch)

	def predictor(self, weight_path):

		self._sub_model.load_weights(weight_path)
		print('Load model from %s!' % weight_path)
		model = self._full_model

		def predict(data):
			print(data)
			data = data.fillna('')
			data[TEXT_COL] = data[TEXT_COL].applymap(pre_treat)
			data[TEXT_COL[1:]] = data[TEXT_COL[1:]].applymap(lambda t: t.split('。'))

			for c in TEXT_COL[1:]:
				data.loc[:, c] = data[c].apply(lambda t: t[:min([len(t), MAX_LENGTH[c]])])

			data[TEXT_COL[1:]] = data[TEXT_COL[1:]].applymap(lambda t: '。'.join(t))
			data['text'] = data[TEXT_COL].apply(join, axis=1)
			tok, seg = zip(*data['text'].apply(get_token()))
			x = [pad_sequences(tok, BERT_LEN, padding='post'),
			     pad_sequences(seg, BERT_LEN, padding='post')]
			y = model.predict(x, verbose=1)
			pred_id = y.argmax(axis=1)
			pred_code = [gics4[g] for g in pred_id]
			return pred_code
		return predict

