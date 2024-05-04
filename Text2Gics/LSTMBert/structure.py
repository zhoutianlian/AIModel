"""
构建模型结构
"""
from tensorflow.keras.layers import *

__all__ = ['c1', 'c2', 'r1', 'r2', 'r1d2', 'cr2', 'c2k5', 'c2m', 'rr']

def c1(shape, activation):
	"""
	(batch_size, time_step=512, vec_dimension=768) 采用一维卷积DCNN
	:param shape:
	:param activation:
	:return:
	"""
	sub_input = Input(shape=shape, name='SUB_INPUT')

	x = Conv1D(filters=256, kernel_size=3, strides=3, padding='same', activation=activation)(sub_input)
	x = BatchNormalization(momentum=0.85)(x)

	x = Conv1D(filters=128, kernel_size=3, strides=3, padding='same', activation=activation)(x)
	x = BatchNormalization(momentum=0.85)(x)

	x = Conv1D(filters=64, kernel_size=3, strides=3, padding='same', activation=activation)(x)
	x = BatchNormalization(momentum=0.85)(x)

	x = Conv1D(filters=32, kernel_size=3, strides=3, padding='same', activation=activation)(x)
	x = BatchNormalization(momentum=0.85)(x)

	x = Flatten(name='FLATTEN')(x)
	sub_output = Dropout(0.2)(x)

	return sub_input, sub_output

def c2(shape, activation):
	"""
	(batch_size, time_step=512, vec_dimension=768, channel=1) 采用二维卷积DCNN
	:param shape:
	:param activation:
	:return:
	"""
	sub_input = Input(shape=shape, name='SUB_INPUT')

	x = Reshape(target_shape=shape + (1,))(sub_input)
	x = Conv2D(filters=32, kernel_size=3, strides=3, padding='same', activation=activation)(x)
	x = BatchNormalization(momentum=0.85)(x)

	x = Conv2D(filters=64, kernel_size=3, strides=3, padding='same', activation=activation)(x)
	x = BatchNormalization(momentum=0.85)(x)

	x = Conv2D(filters=128, kernel_size=3, strides=3, padding='same', activation=activation)(x)
	x = BatchNormalization(momentum=0.85)(x)

	x = Conv2D(filters=256, kernel_size=3, strides=3, padding='same', activation=activation)(x)
	x = BatchNormalization(momentum=0.85)(x)

	x = Flatten(name='FLATTEN')(x)
	sub_output = Dropout(0.2)(x)

	return sub_input, sub_output

def c2k5(shape, activation):
	"""
	(batch_size, time_step=512, vec_dimension=768, channel=1) 采用二维卷积DCNN
	:param shape:
	:param activation:
	:return:
	"""
	sub_input = Input(shape=shape, name='SUB_INPUT')

	x = Reshape(target_shape=shape + (1,))(sub_input)
	x = Conv2D(filters=64, kernel_size=5, strides=5, padding='same', activation=activation)(x)
	x = BatchNormalization(momentum=0.85)(x)

	x = Conv2D(filters=128, kernel_size=5, strides=5, padding='same', activation=activation)(x)
	x = BatchNormalization(momentum=0.85)(x)

	x = Conv2D(filters=256, kernel_size=5, strides=5, padding='same', activation=activation)(x)
	x = BatchNormalization(momentum=0.85)(x)

	x = Flatten(name='FLATTEN')(x)
	sub_output = Dropout(0.2)(x)

	return sub_input, sub_output

def c2m(shape, activation):
	"""
	(batch_size, time_step=512, vec_dimension=768, channel=1) 采用二维卷积DCNN
	:param shape:
	:param activation:
	:return:
	"""
	sub_input = Input(shape=shape, name='SUB_INPUT')

	x = Reshape(target_shape=shape + (1,))(sub_input)
	x = Conv2D(filters=32, kernel_size=3, padding='same', activation=activation)(x)
	x = MaxPooling2D(pool_size=2, strides=2, padding='same')(x)
	x = BatchNormalization(momentum=0.85)(x)

	x = Conv2D(filters=64, kernel_size=3, padding='same', activation=activation)(x)
	x = MaxPooling2D(pool_size=2, strides=2, padding='same')(x)
	x = BatchNormalization(momentum=0.85)(x)

	x = Conv2D(filters=128, kernel_size=3, padding='same', activation=activation)(x)
	x = MaxPooling2D(pool_size=2, strides=2, padding='same')(x)
	x = BatchNormalization(momentum=0.85)(x)

	x = Conv2D(filters=256, kernel_size=3, padding='same', activation=activation)(x)
	x = MaxPooling2D(pool_size=2, strides=2, padding='same')(x)
	x = BatchNormalization(momentum=0.85)(x)

	x = Conv2D(filters=512, kernel_size=3, padding='same', activation=activation)(x)
	x = MaxPooling2D(pool_size=2, strides=2, padding='same')(x)
	x = BatchNormalization(momentum=0.85)(x)

	x = Conv2D(filters=512, kernel_size=3, padding='same', activation=activation)(x)
	x = MaxPooling2D(pool_size=2, strides=2, padding='same')(x)
	x = BatchNormalization(momentum=0.85)(x)

	x = Flatten(name='FLATTEN')(x)
	sub_output = Dropout(0.2)(x)

	return sub_input, sub_output

def r1(shape, activation):
	"""
	(batch_size, time_step=512, vec_dimension=768) 采用单向LSTM=128+浅层DNN
	:param shape:
	:param activation:
	:return:
	"""
	sub_input = Input(shape=shape, name='SUB_INPUT')

	x = LSTM(128, activation='tanh', dropout=0.2)(sub_input)
	x = Flatten()(x)
	x = BatchNormalization(momentum=0.85)(x)
	x = Dense(256, activation=activation)(x)
	sub_output = Dropout(0.2)(x)
	return sub_input, sub_output

def r2(shape, activation):
	"""
	(batch_size, time_step=512, vec_dimension=768) 采用双向LSTM=128*2+浅层DNN=1
	:param shape:
	:param activation:
	:return:
	"""
	sub_input = Input(shape=shape, name='SUB_INPUT')

	x = Bidirectional(LSTM(128, activation='tanh', dropout=0.2))(sub_input)
	x = Flatten()(x)
	x = BatchNormalization(momentum=0.85)(x)
	x = Dense(256, activation=activation)(x)
	sub_output = Dropout(0.2)(x)
	return sub_input, sub_output

def r1d2(shape, activation):
	"""
	(batch_size, time_step=512, vec_dimension=768) 采用双向LSTM=128*2+深层DNN=2
	:param shape:
	:param activation:
	:return:
	"""
	sub_input = Input(shape=shape, name='SUB_INPUT')

	x = LSTM(128, activation='tanh', dropout=0.2)(sub_input)
	x = Flatten()(x)
	x = BatchNormalization(momentum=0.85)(x)
	x = Dense(128, activation=activation)(x)
	x = Dropout(0.2)(x)
	x = Dense(128, activation=activation)(x)
	sub_output = Dropout(0.2)(x)
	return sub_input, sub_output

def cr2(shape, activation):
	"""
	(batch_size, time_step=512, vec_d1=32, vec_d2=24, channel=1) 采用RCNN
	:param shape:
	:param activation:
	:return:
	"""
	sub_input = Input(shape=shape, name='SUB_INPUT')
	h, w = shape[-1] // 24, 24
	x = Reshape(target_shape=(shape[0], h, w, 1))(sub_input)
	x = ConvLSTM2D(filters=32, kernel_size=3, strides=3, padding='same', activation='tanh')(x)
	# 输出为 (batch_size, h, w, channel)
	x = BatchNormalization(momentum=0.85)(x)
	x = Conv2D(filters=64, kernel_size=3, strides=3, padding='same', activation='tanh')(x)
	x = BatchNormalization(momentum=0.85)(x)
	x = Conv2D(filters=128, kernel_size=3, strides=3, padding='same', activation='tanh')(x)
	x = BatchNormalization(momentum=0.85)(x)
	x = Flatten()(x)
	sub_output = Dropout(0.2)(x)
	return sub_input, sub_output

def rr(shape, activation):
	"""
	(batch_size, time_step=512, vec_dim=768)  两层RNN
	:param shape:
	:param activation:
	:return:
	"""
	sub_input = Input(shape=shape, name='SUB_INPUT')
	x = Bidirectional(LSTM(128, activation='tanh', dropout=0.2, recurrent_dropout=0.2, return_sequences=True))(sub_input)
	x = BatchNormalization(momentum=0.85)(x)
	x = Bidirectional(LSTM(128, activation='tanh', dropout=0.2, recurrent_dropout=0.2))(x)
	x = BatchNormalization(momentum=0.85)(x)
	x = Flatten()(x)
	x = Dense(256, activation=activation)(x)
	sub_output = Dropout(0.2)(x)
	return sub_input, sub_output
