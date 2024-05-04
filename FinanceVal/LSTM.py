"""
构建LSTM模型

"""
import warnings
warnings.filterwarnings("ignore")

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

def get_model(optimizer, loss, metrics):
	# Model
	x_val_input = Input((10, 4), name='InputVal')
	x_lab_input = Input((10, 2), name='InputLab')

	x1 = Embedding(11, 3)(x_lab_input[..., 0])
	x2 = Embedding(10, 4)(x_lab_input[..., 1])
	x = Concatenate()([x_val_input, x1, x2])

	x = [LSTM(256, activation='tanh')(Attention(dropout=0.1, name='DotAtt%0d' % i)([x, x])) for i in range(8)]
	x = Concatenate()(x)
	x = BatchNormalization(momentum=0.85, name='BN01')(x)
	x = Dense(128, name='Dense01')(x)
	x = LeakyReLU(name='LR01')(x)

	x = BatchNormalization(momentum=0.85, name='BN02')(x)
	x = Dense(64, name='Dense02')(x)
	x = LeakyReLU(name='LR02')(x)

	x = BatchNormalization(momentum=0.85, name='BN03')(x)
	x = Dense(6, activation='relu', name='Output')(x)
	x = Reshape((2, 3))(x)

	model = Model([x_val_input, x_lab_input], x)
	model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
	return model