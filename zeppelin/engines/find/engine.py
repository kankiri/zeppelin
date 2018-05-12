import numpy as np

from zeppelin import Engine as BaseEngine


class Engine(BaseEngine):
	def __init__(self, agents, dimensions=2, factor=1):
		super().__init__(agents)
		self.agent_name = list(agents)[0]
		self.target = np.array([14] * dimensions)
		self.factor = factor
		
		self.position = None
		self.distance = None
		self.time = None
	
	def reset(self):
		self.position = np.random.normal(self.target, 20/len(self.target))
		self.distance = self._get_distance()
		self.time = 0
		return {self.agent_name: {'position': self.position}}
		
	def step(self, action):
		action = action[self.agent_name]['action']
		self.position += self._direction(action)
		distance = self._get_distance()
		reward = (self.distance - distance) * self.factor
		self.distance = distance
		done = False
		
		if self.distance < 0.5:
			reward += 20
			done = 'WN'
		elif self.time > 1000:
			reward -= 0
			done = 'TU'
		
		self.time += 1
		if done:
			self.reset()
		return {self.agent_name: {
			'position': self.position,
			'reward': reward,
			'done': done
		}}
		
	def _direction(self, action):
		result = np.zeros(self.target.shape)
		value = 0.2 if action % 2 == 0 else -0.2
		result[int(action/2)] = value
		return result
	
	def _get_distance(self):
		return np.linalg.norm(self.position - self.target)
