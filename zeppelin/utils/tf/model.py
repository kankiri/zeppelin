import tensorflow as tf


class Model:
	def __init__(self, input_shapes, output_shapes):
		self.input_shapes = input_shapes
		self.output_shapes = output_shapes
		
		graph = tf.Graph()
		with graph.as_default():
			self.inputs, self.outputs = self._network()
			self.weights = graph.get_collection('trainable_variables')
			self.references, self.updates = self._updates(self.outputs)
			self.gradients, self.apply = self._training()
			
		self.session = tf.Session(graph=graph)
		self.session.run(tf.variables_initializer(graph.get_collection('variables')))
	
	def predict(self, inputs):
		feed = dict(zip(self.inputs, inputs))
		return self.session.run(self.outputs, feed_dict=feed)
	
	def compute(self, inputs, references):
		feed = dict(zip(self.inputs, inputs))
		feed.update(zip(self.references, references))
		return self.session.run(self.updates, feed_dict=feed)
	
	def apply(self, gradients):
		feed = dict(zip(self.gradients, gradients))
		self.session.run(self.apply, feed_dict=feed)
	
	def fit(self, inputs, references):
		feed = dict(zip(self.inputs, inputs))
		feed.update(zip(self.references, references))
		gradients = self.session.run(self.updates, feed_dict=feed)
		feed = dict(zip(self.gradients, gradients))
		self.session.run(self.apply, feed_dict=feed)
	
	def get_parameters(self):
		return self.session.run(self.weights)
	
	def set_parameters(self, values):
		for weight, value in zip(self.weights, values):
			weight.load(value, self.session)
	
	def _network(self):
		return None, None
	
	def _updates(self, outputs):
		return None, None
	
	def _training(self):
		return None, None
