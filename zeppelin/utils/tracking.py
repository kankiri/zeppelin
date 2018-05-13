from functools import partial
import types


def add(agent, function):
	react = agent.react
	wrapped = partial(function, react)
	agent.react = types.MethodType(wrapped, agent)
	return agent
