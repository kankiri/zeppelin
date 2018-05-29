from zeppelin.utils import Agent as BaseAgent


class Agent(BaseAgent):
	def __init__(self, name):
		super().__init__(name)
		self.actions = '0333000031000210'
	
	def react(self, position, reward=0, done=None):
		action = int(self.actions[position])
		super().react(reward, done)
		return {'action': action}
