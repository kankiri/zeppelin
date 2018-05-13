import numpy as np

from zeppelin.utils import Transitions
from .models import Model
from zeppelin import Agent as BaseAgent


class Agent(BaseAgent):
	def __init__(self, name, dimensions, gamma=0.95, epsilon=1, decay=1-1e-3):
		super().__init__(name)
		self.gamma = gamma
		self.epsilon = epsilon
		self.decay = decay
		
		self.model = Model(((dimensions,),), ((dimensions*2,),), [7])
		self.episode = 0
		self.memory = Transitions(
			cause_keys=['positions', 'actions'],
			effect_keys=['rewards', 'dones', 'outcomes'],
			extra_keys=['perf'],
			maxlen=10
		)
	
	def react(self, position, reward=0, done=False):
		action = self.respond(position)
		self.memory.store(position.copy(), action, reward, done, position.copy(), perf=reward)
		if self.age != 0:
			self.learn()
		if done:
			self.epsilon *= self.decay
			self.episode += 1
		self.age += 1
		return {'action': action}
	
	def respond(self, position):
		if np.random.rand() < self.epsilon:
			return np.random.randint(0,  *self.model.output_shapes[0])
		else:
			prediction = self.model.predict([[position]])[0][0]
			return np.argmax(prediction)
	
	def learn(self):
		position, action, reward, done, outcome = self.memory[-2]
		past_value_prediction = self.model.predict([[position]])[0][0]
		future_value_prediction = [0] if done else self.model.predict([[outcome]])[0][0]
		future_value_prediction = np.max(future_value_prediction)

		target = past_value_prediction
		target[action] = reward + self.gamma * future_value_prediction
		
		self.model.fit([[position]], [[target]])
