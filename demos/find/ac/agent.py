import numpy as np

from zeppelin.utils import Transitions
from .models import ActorModel, CriticModel
from zeppelin import Agent as BaseAgent
from zeppelin.utils import discount


class Agent(BaseAgent):
	def __init__(self, name, dimensions, gamma=0.95, epsilon=1, decay=1-1e-3):
		super().__init__(name)
		self.gamma = gamma
		self.epsilon = epsilon
		self.decay = decay
		
		self.actor_model = ActorModel(((dimensions,),), ((dimensions*2,),), [7])
		self.critic_model = CriticModel(((dimensions,),), ((1,),), [7])
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
			return np.random.randint(0,  *self.actor_model.output_shapes[0])
		else:
			prediction = self.actor_model.predict([[position]])[0][0]
			choice = np.random.choice(prediction, p=prediction)
			return np.argmax(prediction == choice)
	
	def learn(self, number=1):
		positions, actions, rewards, dones, outcomes = self.memory[-(number+1):-1]
		past_value_predictions = self.critic_model.predict([positions])[0]
		future_value_prediction = [0] if dones[-1] else self.critic_model.predict([outcomes[-1:]])[0][0]
		
		targets = discount(np.concatenate((rewards, future_value_prediction)), self.gamma)[:-1].reshape(-1, 1)
		errors = targets - past_value_predictions
		
		self.actor_model.fit([positions], [actions.reshape(-1, 1), errors])
		self.critic_model.fit([positions], [targets])
