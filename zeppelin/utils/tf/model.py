import tensorflow as tf


class Model:
	def __init__(self, input_shapes, output_shapes):
		self.input_shapes = input_shapes
		self.output_shapes = output_shapes
		
		graph = tf.Graph()
		with graph.as_default():
			self.inputs, self.outputs = self._network()
			self.weights = graph.get_collection('trainable_variables')
			self.references, self.loss, self.gradients_out = self._loss()
			self.gradients_in, self.apply, self.minimize = self._training()
			
		self.session = tf.Session(graph=graph)
		self.session.run(tf.variables_initializer(graph.get_collection('variables')))
	
	def predict(self, inputs):
		feed = dict(zip(self.inputs, inputs))
		return self.session.run(self.outputs, feed_dict=feed)
	
	def compute(self, inputs, references):
		feed = dict(zip(self.inputs, inputs))
		feed.update(zip(self.references, references))
		return self.session.run(self.gradients_out, feed_dict=feed)
	
	def apply(self, gradients):
		feed = dict(zip(self.gradients_in, gradients))
		self.session.run(self.apply, feed_dict=feed)
	
	def fit(self, inputs, references):
		feed = dict(zip(self.inputs, inputs))
		feed.update(zip(self.references, references))
		self.session.run(self.minimize, feed_dict=feed)
	
	def get_parameters(self):
		return self.session.run(self.weights)
	
	def set_parameters(self, values):
		for weight, value in zip(self.weights, values):
			weight.load(value, self.session)
	
	def _network(self):
		return None, None
	
	def _loss(self):
		return None, None, None
	
	def _training(self):
		return None, None, None
