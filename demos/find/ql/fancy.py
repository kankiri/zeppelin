import numpy as np
from zeppelin.utils import Agent as BaseAgent, Transitions

from zeppelin.utils.tf.ql import Model


class Agent(BaseAgent):
	def __init__(self, name, dimensions, batch=10, gamma=0.95, epsilon=1, decay=1-1e-4, frequency=1000):
		super().__init__(name)
		self.batch = batch
		self.gamma = gamma
		self.epsilon = epsilon
		self.decay = decay
		self.frequency = frequency

		self.model = Model([[dimensions]], [[dimensions * 2]], [7])
		self.target = Model([[dimensions]], [[dimensions * 2]], [7])
		self.memory = Transitions(['positions', 'actions'], ['rewards', 'dones', 'outcomes'])

	def react(self, position, time, reward=0, done=None):
		action = self.respond(position)
		self.memory.store(position.copy(), action, reward, bool(done), position.copy())
		if self.age % self.batch == (self.batch - 1) or done:
			self.learn(self.batch)
		if self.age % self.frequency == (self.frequency - 1):
			self.target.set_parameters(self.model.get_parameters())
		if done:
			self.epsilon *= self.decay
		super().react(reward, done)
		return {'action': action}

	def respond(self, position):
		if np.random.rand() < self.epsilon:
			return np.random.randint(0, *self.model.output_shapes[0])
		else:
			prediction = self.model.predict([[position]])[0][0]
			return np.argmax(prediction)

	def learn(self, number=1):
		positions, actions, rewards, dones, outcomes = self.memory.shuffled(number)
		if len(positions) >= 1:
			past_value_predictions = self.model.predict([positions])[0]
			future_value_predictions = np.select([~dones.reshape(-1, 1)], [self.target.predict([outcomes])[0]])
			future_value_predictions = np.max(future_value_predictions, axis=1)

			targets = past_value_predictions
			targets[range(len(targets)), actions] = rewards + self.gamma * future_value_predictions

			self.model.fit([positions], [targets])
