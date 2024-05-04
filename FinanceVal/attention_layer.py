"""
自定义神经网络层
- AttentionLayer 注意力机制的层
- ConditionalInstanceNormalization 条件实例归一化层
"""
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K


class AttentionLayer(Layer):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	def build(self, input_shape):
		assert len(input_shape) == 3
		# W.shape = (time_steps, time_steps)
		self.W = self.add_weight(name='att_weight',
		                         shape=(input_shape[1], input_shape[1]),
		                         initializer='uniform',
		                         trainable=True)
		self.b = self.add_weight(name='att_bias',
		                         shape=(input_shape[1],),
		                         initializer='uniform',
		                         trainable=True)
		super().build(input_shape)

	def call(self, inputs, **kwargs):
		# inputs.shape = (batch_size, time_steps, seq_len)
		x = K.permute_dimensions(inputs, (0, 2, 1))
		# x.shape = (batch_size, seq_len, time_steps)
		a = K.softmax(K.tanh(K.dot(x, self.W) + self.b))
		outputs = K.permute_dimensions(a * x, (0, 2, 1))
		outputs = K.sum(outputs, axis=1)
		return outputs

	def compute_output_shape(self, input_shape):
		return input_shape[0], input_shape[2]


# class ConditionalInstanceNormalization(Layer):
# 	def __init__(self,
# 	             input_dim: int,
# 	             embeddings_initializer='uniform',
# 	             embeddings_regularizer=None,
# 	             activity_regularizer=None,
# 	             embeddings_constraint=None,
# 	             mask_zero=False,
# 	             **kwargs):
# 		assert input_dim > 1
# 		self.input_dim = input_dim
# 		self.embeddings_initializer = initializers.get(embeddings_initializer)
# 		self.embeddings_regularizer = regularizers.get(embeddings_regularizer)
# 		self.activity_regularizer = regularizers.get(activity_regularizer)
# 		self.embeddings_constraint = constraints.get(embeddings_constraint)
# 		self.mask_zero = mask_zero
# 		self.supports_masking = mask_zero
# 		super().__init__(**kwargs)
#
# 	def build(self, input_shape):
# 		assert len(input_shape) > 1
# 		if context.executing_eagerly() and context.context().num_gpus():
# 			with ops.device('cpu:0'):
# 				self.mu_sigma = self.add_weight(
# 					shape=(self.input_dim, 2),
# 					initializer=self.embeddings_initializer,
# 					name='embeddings',
# 					regularizer=self.embeddings_regularizer,
# 					constraint=self.embeddings_constraint)
# 		else:
# 			self.mu_sigma = self.add_weight(
# 				shape=(self.input_dim, 2),
# 				initializer=self.embeddings_initializer,
# 				name='embeddings',
# 				regularizer=self.embeddings_regularizer,
# 				constraint=self.embeddings_constraint)
#
# 		self.built = True
#
# 	def compute_mask(self, inputs, mask=None):
# 		if not self.mask_zero:
# 			return None
# 		return math_ops.not_equal(inputs, 0)
#
# 	@tf_utils.shape_type_conversion
# 	def compute_output_shape(self, input_shape):
# 		if self.input_length is None:
# 			return input_shape + (self.output_dim,)
# 		else:
# 			# input_length can be tuple if input is 3D or higher
# 			if isinstance(self.input_length, (list, tuple)):
# 				in_lens = list(self.input_length)
# 			else:
# 				in_lens = [self.input_length]
# 		if len(in_lens) != len(input_shape) - 1:
# 			raise ValueError('"input_length" is %s, '
# 			                 'but received input has shape %s' % (str(
# 				self.input_length), str(input_shape)))
# 		else:
# 			for i, (s1, s2) in enumerate(zip(in_lens, input_shape[1:])):
# 				if s1 is not None and s2 is not None and s1 != s2:
# 					raise ValueError('"input_length" is %s, but received input has shape %s' %
# 					                 (str(self.input_length), str(input_shape)))
# 				elif s1 is None:
# 					in_lens[i] = s2
# 		return (input_shape[0],) + tuple(in_lens) + (self.output_dim,)
#
# 	def call(self, inputs, **kwargs):
# 		assert len(inputs) == 2
# 		x, _id = inputs
# 		dtype = K.dtype(_id)
# 		if dtype != 'int32' and dtype != 'int64':
# 			_id = K.cast(_id, 'int32')
# 		if isinstance(self.mu_sigma, sharded_variable.ShardedVariable):
# 			param = embedding_ops.embedding_lookup_v2(self.mu_sigma.variables, _id)
# 		else:
# 			param = embedding_ops.embedding_lookup_v2(self.mu_sigma, _id)
#
# 		mu, sigma = param[:, :1], param[:, 1:]
# 		return (x - mu) / sigma
#
# 	def get_config(self):
# 		config = {
# 			'input_dim': self.input_dim,
# 			'embeddings_initializer': initializers.serialize(self.embeddings_initializer),
# 			'embeddings_regularizer': regularizers.serialize(self.embeddings_regularizer),
# 			'activity_regularizer': regularizers.serialize(self.activity_regularizer),
# 			'embeddings_constraint': constraints.serialize(self.embeddings_constraint),
# 			'mask_zero': self.mask_zero,
# 		}
# 		base_config = super().get_config()
# 		return dict(list(base_config.items()) + list(config.items()))
