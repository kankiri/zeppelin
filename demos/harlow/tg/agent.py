from random import randrange

from zeppelin import Agent as BaseAgent
from zeppelin.utils import Transitions


class Agent(BaseAgent):
	def __init__(self, name):
		super().__init__(name)
		self.memory = Transitions(['chosen'], extra_keys=['perf'])
	
	def react(self, numbers, reward=0, done=None):
		action = self.respond(numbers, reward, done)
		self.memory.store(numbers[action], perf=reward)
		if done:
			self.memory.forget()
		self.age += 1
		return {'action': action}
	
	def respond(self, numbers, reward, done):
		if done or not self.age:
			return randrange(2)
		chosen, = self.memory[-1]
		if reward > 0:
			return numbers.index(chosen)
		return 1 - numbers.index(chosen)
