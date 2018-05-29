import numpy as np
from zeppelin.utils import Agent as BaseAgent, discount, Transitions

from .models import Model


class Agent(BaseAgent):
	def __init__(self, name, gamma=0.95, epsilon=1, decay=1-1e-4):
		super().__init__(name)
		self.gamma = gamma
		self.epsilon = epsilon
		self.decay = decay
		
		self.model = Model(((16,),), ((4,),))
		self.memory = Transitions(['positions', 'actions'], ['rewards', 'dones', 'outcomes'])
	
	def react(self, position, reward=0, done=None):
		zeros = np.zeros(16); zeros[position] = 1; position = zeros
		action = self.respond(position)
		self.memory.store(position.copy(), action, reward, bool(done), position.copy())
		if done:
			self.learn()
			self.memory.forget()
			self.epsilon *= self.decay
		super().react(reward, done)
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
			# advantages = (advantages - np.mean(advantages)) / (np.std(advantages) or 1e-9)
			self.model.fit([positions], [actions.reshape(-1, 1), advantages])
