import numpy as np

from zeppelin import Engine as BaseEngine


class Engine(BaseEngine):
	def __init__(self, agents, dimensions=2, maxtime=1000, step_factor=1, win_factor=0, time_factor=0):
		super().__init__(agents)
		self.agent_name = list(agents)[0]
		self.center = np.array([0] * dimensions)
		self.maxtime = maxtime
		self.step_factor = step_factor
		self.win_factor = win_factor
		self.time_factor = time_factor
		
		self.target = None
		self.position = None
		self.distance = None
		self.time = None
	
	def reset(self):
		self.target = np.random.normal(self.center, 20/len(self.center))
		self.position = np.random.normal(self.center, 20/len(self.center))
		self.distance = self._get_distance()
		self.time = 0
		return {self.agent_name: {
			'position': self.position,
			'direction': 0,
			'time': self.time
		}}
	
	def step(self, action):
		action = action[self.agent_name]['action']
		self.position += self._direction(action)
		distance = self._get_distance()
		direction = (self.distance - distance)
		reward = direction * self.step_factor
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
			'direction': direction,
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
