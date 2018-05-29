from random import choices, seed

from zeppelin import Agent as BaseAgent
from zeppelin.utils import Transitions


class Agent(BaseAgent):
	def __init__(self, name, dimensions=5, length=10):
		super().__init__(name)
		seed(0)
		self.sequence = choices(range(dimensions), k=length)
		self.index = None
		self.memory = Transitions(['dummy'], extra_keys=['perf'])
	
	def react(self, reward=0, done=None):
		action = self.respond(done)
		self.memory.store(perf=reward)
		if done:
			self.memory.forget()
		self.age += 1
		return {'action': action}
	
	def respond(self, done):
		if done or not self.age:
			self.index = -1
		self.index += 1
		return self.sequence[self.index]
