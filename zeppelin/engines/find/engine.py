import numpy as np

from . import config
from zeppelin import Engine as BaseEngine


class Engine(BaseEngine):
	def __init__(self, agents, dimensions=2, maxtime=1000, step_factor=1, win_factor=0, time_factor=0):
		super().__init__(agents)
		self.observation_shapes = config.observation_shapes
		self.observation_shapes['position']['shape'] = (dimensions,)
		self.action_shapes = config.action_shapes
		self.action_shapes['action']['max'] = (dimensions*2,)

		self.agent_name = list(agents)[0]
		self.target = np.array([14] * dimensions)
		self.maxtime = maxtime
		self.step_factor = step_factor
		self.win_factor = win_factor
		self.time_factor = time_factor
		
		self.position = None
		self.distance = None
		self.time = None
	
	def reset(self):
		self.position = np.random.normal(self.target, 20/len(self.target))
		self.distance = self._get_distance()
		self.time = 0
		return {self.agent_name: {
			'position': self.position,
			'time': self.time
		}}
	
	def step(self, actions):
		action = actions[self.agent_name]['action']
		self.position += self._direction(action)
		distance = self._get_distance()
		reward = (self.distance - distance) * self.step_factor
		self.distance = distance
		done = False
		
		if self.distance < 0.5:
			reward += 10 * self.win_factor
			done = 'WN'
		elif self.time > self.maxtime:
			reward -= 10 * self.time_factor
			done = 'TU'
		
		self.time += 1
		if done:
			self.reset()
		return {self.agent_name: {
			'position': self.position,
			'time': self.time,
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
