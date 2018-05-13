import numpy as np
import tensorflow as tf

from zeppelin.utils.tf import Model as BaseModel


class Model(BaseModel):
	def __init__(self, input_shapes, output_shapes, hidden_layers=[], learning_rate=0.001):
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
		outputs = tf.layers.dense(inputs=connection, units=self.output_shapes[0][0],
			kernel_initializer=tf.keras.initializers.he_normal(), kernel_regularizer=tf.contrib.layers.l2_regularizer)
		return inputs, [outputs]
	
	def _updates(self, outputs):
		targets = [tf.placeholder(shape=(None, *shape), dtype=tf.float32, name='Targets_{}'.format(i))
			for i, shape in enumerate(self.output_shapes)]
		
		loss = tf.losses.mean_squared_error(predictions=outputs[0], labels=targets[0])
		updates = tf.gradients(loss, self.weights)
		return targets, updates
	
	def _training(self):
		gradients = [tf.placeholder(shape=variable.shape, dtype=tf.float32, name='Gradients_{}'.format(i))
			for i, variable in enumerate(self.weights)]
		
		clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
		optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
		apply = optimizer.apply_gradients(zip(clipped_gradients, self.weights))
		return gradients, apply
