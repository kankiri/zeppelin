class Agent:
	def __init__(self, name):
		self.name = name
		self.age = 0
	
	def react(self):
		self.age += 1


class ConnectedAgent(Agent):
	def send(self):
		pass
	
	def receive(self):
		pass
	
	def publish(self):
		pass
	
	def read(self):
		pass
