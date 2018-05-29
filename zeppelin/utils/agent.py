from zeppelin import Agent as BaseAgent


class Agent(BaseAgent):
	def __init__(self, name, track=10000):
		super().__init__(name)
		self.track = track
		self.episode = 0
		self.performance = 0

	def react(self, reward=0, done=None):
		self.performance += reward
		if self.age % self.track == self.track-1:
			with open(f'{self.name}.csv', 'a') as file:
				file.write(f'{self.performance},{self.episode}\n')
			self.performance = 0
		if done:
			self.episode += 1
		self.age += 1
