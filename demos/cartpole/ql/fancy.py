import numpy as np
from zeppelin.utils import Agent as BaseAgent, Transitions

from zeppelin.utils.tf.ql import Model


class Agent(BaseAgent):
	def __init__(self, name, batch=10, gamma=0.95, epsilon=1, decay=1-1e-4, frequency=1000):
		super().__init__(name)
		self.batch = batch
		self.gamma = gamma
		self.epsilon = epsilon
		self.decay = decay
		self.frequency = frequency

		self.model = Model([[4]], [[2]], [15, 15])
		self.target = Model([[4]], [[2]], [15, 15])
		self.memory = Transitions(['states', 'actions'], ['rewards', 'dones', 'outcomes'])

	def react(self, state, reward=0, done=None):
		action = self.respond(state)
		self.memory.store(state.copy(), action, reward, bool(done), state.copy())
		if self.age % self.batch == (self.batch - 1) or done:
			self.learn(self.batch)
		if self.age % self.frequency == (self.frequency - 1):
			self.target.set_parameters(self.model.get_parameters())
		if done:
			self.epsilon *= self.decay
		super().react(reward, done)
		return {'action': action}

	def respond(self, state):
		if np.random.rand() < self.epsilon:
			return np.random.randint(0, *self.model.output_shapes[0])
		else:
			prediction = self.model.predict([[state]])[0][0]
			return np.argmax(prediction)

	def learn(self, number=1):
		states, actions, rewards, dones, outcomes = self.memory.shuffled(number)
		if len(states) >= 1:
			past_value_predictions = self.model.predict([states])[0]
			future_value_predictions = np.select([~dones.reshape(-1, 1)], [self.target.predict([outcomes])[0]])
			future_value_predictions = np.max(future_value_predictions, axis=1)

			targets = past_value_predictions
			targets[range(len(targets)), actions] = rewards + self.gamma * future_value_predictions

			self.model.fit([states], [targets])
