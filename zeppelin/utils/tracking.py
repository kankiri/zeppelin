from functools import partial
import types

from zeppelin import Agent, World


def add(o, method):
	if isinstance(o, Agent):
		react = o.react
		wrapped = partial(method, react)
		o.react = types.MethodType(wrapped, o)
		return o
	elif isinstance(o, World):
		step = o.step
		wrapped = partial(method, step)
		o.step = types.MethodType(wrapped, o)
		return o
