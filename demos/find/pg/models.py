import numpy as np
import tensorflow as tf

from zeppelin.utils.tf import Model as BaseModel


class Model(BaseModel):
	def __init__(self, input_shapes, output_shapes, hidden_layers=[], learning_rate=0.0005):
		self.hidden_layers = hidden_layers
		self.learning_rate = learning_rate
		super().__init__(input_shapes, output_shapes)
	
	def _network(self):
		inputs = [tf.placeholder(shape=(None, *shape), dtype=tf.float32, name='Inputs_{}'.format(i))
			for i, shape in enumerate(self.input_shapes)]
		
		connection = inputs[0]
		for units in self.hidden_layers:
			connection = tf.layers.dense(inputs=connection, units=units, activation=tf.nn.elu,
				kernel_initializer=tf.keras.initializers.he_normal(), kernel_regularizer=tf.contrib.layers.l2_regularizer)
		outputs = tf.layers.dense(inputs=connection, units=self.output_shapes[0][0], activation=tf.nn.softmax,
			kernel_initializer=tf.keras.initializers.he_normal())
		return inputs, [outputs]
	
	def _updates(self, outputs):
		actions = tf.placeholder(shape=(None, 1), dtype=tf.int32, name='Actions')
		advantages = tf.placeholder(shape=(None, 1), dtype=tf.float32, name='Advantages')
		
		onehot = tf.one_hot(actions, self.output_shapes[0][0], dtype=tf.float32)
		action_values = tf.reduce_sum(outputs[0] * tf.squeeze(onehot), axis=1, keepdims=True)
		action_values = tf.clip_by_value(action_values, 1e-9, 1-1e-9)
		loss = -tf.reduce_sum(advantages * tf.log(action_values))
		updates = tf.gradients(loss, self.weights)
		return [actions, advantages], updates
	
	def _training(self):
		gradients = [tf.placeholder(shape=variable.shape, dtype=tf.float32, name='Gradients_{}'.format(i))
			for i, variable in enumerate(self.weights)]
		
		clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
		optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
		apply = optimizer.apply_gradients(zip(clipped_gradients, self.weights))
		return gradients, apply
