import numpy as np
from zeppelin.utils import Agent as BaseAgent, discount, Transitions

from .models import ActorModel, CriticModel


class Agent(BaseAgent):
	def __init__(self, name, dimensions, batch=10, gamma=0.95, epsilon=1, decay=1-1e-4):
		super().__init__(name)
		self.batch = batch
		self.gamma = gamma
		self.epsilon = epsilon
		self.decay = decay
		
		self.actor_model = ActorModel(((dimensions,),), ((dimensions*2,),))
		self.critic_model = CriticModel(((dimensions,),), ((1,),), [7])
		self.memory = Transitions(['positions', 'actions'], ['rewards', 'dones', 'outcomes'])
	
	def react(self, position, time, reward=0, done=None):
		action = self.respond(position)
		self.memory.store(position.copy(), action, reward, bool(done), position.copy())
		if self.age % self.batch == (self.batch - 1) or done:
			self.learn(self.batch)
		if done:
			self.memory.forget()
			self.epsilon *= self.decay
		super().react(reward, done)
		return {'action': action}
	
	def respond(self, position):
		if np.random.rand() < self.epsilon:
			return np.random.randint(0, *self.actor_model.output_shapes[0])
		else:
			prediction = self.actor_model.predict([[position]])[0][0]
			choice = np.random.choice(prediction, p=prediction)
			return np.argmax(prediction == choice)
	
	def learn(self, number=1):
		positions, actions, rewards, dones, outcomes = self.memory[-(number+1):-1]
		if len(positions) >= 1:
			past_value_predictions = self.critic_model.predict([positions])[0]
			future_value_prediction = [0] if dones[-1] else self.critic_model.predict([outcomes[-1:]])[0][0]
		
			targets = discount(np.concatenate((rewards, future_value_prediction)), self.gamma)[:-1]
			targets = targets.reshape(-1, 1)
			advantages = targets - past_value_predictions
		
			self.actor_model.fit([positions], [actions.reshape(-1, 1), advantages])
			self.critic_model.fit([positions], [targets])
