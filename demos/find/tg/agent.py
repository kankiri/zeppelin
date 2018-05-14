import numpy as np
import tensorflow as tf

from zeppelin import Agent as BaseAgent
from zeppelin.utils import Transitions
from zeppelin.utils.tf import Model as BaseModel


class Model(BaseModel):
	def _network(self):
		inputs = [tf.placeholder(shape=(None, *shape), dtype=tf.float32, name='Inputs_{}'.format(i))
			for i, shape in enumerate(self.input_shapes)]
		outputs = tf.layers.dense(inputs=inputs[0], units=self.output_shapes[0][0], activation=tf.nn.softmax)
		return inputs, [outputs]


class Agent(BaseAgent):
	def __init__(self, name, dimensions):
		super().__init__(name)
		
		self.model = Model(((dimensions,),), ((dimensions*2,),))
		self.memory = Transitions(['dummy'], extra_keys=['perf'])
		
		weights = np.zeros((dimensions, dimensions * 2))
		for i in range(dimensions):
			weights[i][2*i] = -1.
			weights[i][2*i+1] = 1.
		self.model.set_parameters([
			weights,
			[14., -14.] * dimensions
		])
	
	def react(self, position, time, reward=0, done=False):
		prediction = self.model.predict([position[np.newaxis]])[0][0]
		choice = np.random.choice(prediction, p=prediction)
		action = np.argmax(prediction == choice)
		self.memory.store(perf=reward)
		self.age += 1
		return {'action': action}
