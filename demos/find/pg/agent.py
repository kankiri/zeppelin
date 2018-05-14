import numpy as np

from zeppelin.utils import Transitions
from .models import Model
from zeppelin import Agent as BaseAgent
from zeppelin.utils import discount


class Agent(BaseAgent):
	def __init__(self, name, dimensions, gamma=0.95, epsilon=1, decay=1-1e-4):
		super().__init__(name)
		self.gamma = gamma
		self.epsilon = epsilon
		self.decay = decay
		
		self.model = Model(((dimensions,),), ((dimensions*2,),))
		self.episode = 0
		self.memory = Transitions(
			cause_keys=['positions', 'actions'],
			effect_keys=['rewards', 'dones', 'outcomes'],
			extra_keys=['perf']
		)
	
	def react(self, position, reward=0, done=False):
		action = self.respond(position)
		self.memory.store(position.copy(), action, reward, done, position.copy(), perf=reward)
		if done:
			self.learn()
			self.memory.forget()
			self.epsilon *= self.decay
			self.episode += 1
		self.age += 1
		return {'action': action}
	
	def respond(self, position):
		if np.random.rand() < self.epsilon:
			return np.random.randint(0, *self.model.output_shapes[0])
		else:
			prediction = self.model.predict([[position]])[0][0]
			choice = np.random.choice(prediction, p=prediction)
			return np.argmax(prediction == choice)
	
	def learn(self):
		positions, actions, rewards, dones, outcomes = self.memory[:-1]
		if len(positions) >= 1:
			advantages = discount(rewards, self.gamma).reshape(-1, 1)
			self.model.fit([positions], [actions.reshape(-1, 1), advantages])
