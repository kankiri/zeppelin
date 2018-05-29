import numpy as np
from zeppelin.utils import Agent as BaseAgent, discount, Transitions

from .models import Model


class Agent(BaseAgent):
	def __init__(self, name, gamma=0.95, epsilon=1, decay=1-1e-4):
		super().__init__(name)
		self.gamma = gamma
		self.epsilon = epsilon
		self.decay = decay
		
		self.model = Model(((2,),), ((3,),), [7])
		self.memory = Transitions(['states', 'actions'], ['rewards', 'dones', 'outcomes'])
	
	def react(self, state, reward=0, done=None):
		action = self.respond(state)
		self.memory.store(state.copy(), action, reward, bool(done), state.copy())
		if done:
			self.learn()
			self.memory.forget()
			self.epsilon *= self.decay
		super().react(reward, done)
		return {'action': action}
	
	def respond(self, state):
		if np.random.rand() < self.epsilon:
			return np.random.randint(0, *self.model.output_shapes[0])
		else:
			prediction = self.model.predict([[state]])[0][0]
			choice = np.random.choice(prediction, p=prediction)
			return np.argmax(prediction == choice)
	
	def learn(self):
		states, actions, rewards, dones, outcomes = self.memory[:-1]
		if len(states) >= 1:
			advantages = discount(rewards, self.gamma).reshape(-1, 1)
			# advantages = (advantages - np.mean(advantages)) / (np.std(advantages) or 1e-9)
			self.model.fit([states], [actions.reshape(-1, 1), advantages])
