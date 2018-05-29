import gym

from . import config
from zeppelin import Engine as BaseEngine


class Engine(BaseEngine):
	def __init__(self, agents):
		super().__init__(agents)
		self.observation_shapes = config.observation_shapes
		self.action_shapes = config.action_shapes

		self.agent_name = list(agents)[0]
		self.environment = gym.make('FrozenLake-v0')
	
	def reset(self):
		observations = self.environment.reset()
		return {self.agent_name: {'position': observations}}
		
	def step(self, actions):
		action = actions[self.agent_name]['action']
		observations, reward, done, _ = self.environment.step(action)
		result = {self.agent_name: {
			'position': observations,
			'reward': reward,
			'done': done
		}}
		if done:
			result[self.agent_name].update(self.reset()[self.agent_name])
		return result
