from random import randrange, sample

from zeppelin import Engine as BaseEngine


class Engine(BaseEngine):
	def __init__(self, agents):
		super().__init__(agents)
		self.numbers = None
		self.last = None
		self.favorite = None
	
	def reset(self):
		self.numbers = randrange(100)
		self.numbers = (self.numbers, self.numbers+1)
		self.numbers = [sample(self.numbers, k=2) for i in range(6)]
		self.last = self.numbers.pop()
		self.favorite = self.last[randrange(2)]
		return {name: {
			'numbers': self.last
		} for name in self.agents}
	
	def step(self, actions):
		reward = {name: self.score(actions[name]['action'])
			for name in self.agents}
		done = not bool(self.numbers)
		if done:
			self.reset()
		else:
			self.last = self.numbers.pop()
		return {name: {
			'numbers': self.last,
			'reward': reward[name],
			'done': done
		} for name in self.agents}
	
	def score(self, action):
		if self.last[action] == self.favorite:
			return 1
		return 0
