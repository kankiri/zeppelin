from universe import Engine as BaseEngine

import gym
import numpy as np


class Engine(BaseEngine):
	def __init__(self, agents):
		super().__init__(agents)
		self.agent_name = list(agents)[0]
		self.environment = gym.make('CartPole-v0')
	
	def reset(self):
		observations = self.environment.reset()
		return {self.agent_name: self._to_dict(observations)}
		
	def step(self, action):
		action = action[self.agent_name]['action']
		observations, reward, done, _ = self.environment.step(action)
		result = {
			self.agent_name: {**self._to_dict(observations), **{
				'reward': reward,
				'done': done
			}}
		}
		if done:
			result[self.agent_name].update(self.reset()[self.agent_name])
		return result
		
	def _to_dict(self, observations):
		return {'state': np.array([
			observations[0],
			observations[1],
			observations[2],
			observations[3]
		])}
