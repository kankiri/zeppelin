try:
	import zeppelin
except ImportError:
	from os.path import abspath, join, pardir
	import sys
	sys.path.append(abspath(join(sys.path[0], pardir)))

from zeppelin.utils.school import School

from find import ql as find_ql, tg as find_tg
from frozenlake import ql as fl_ql, tg as fl_tg
from cartpole import ql as cp_ql


if __name__ == '__main__':
	curriculum = {
		'find-basic-rw': {
			'engine': 'find', 'args': [2],
			'agents': [find_ql.BasicAgent('find-basic-rw-agent-2', 2, gamma=0)]
		},
		'find-basic-ql': {
			'engine': 'find', 'args': [2],
			'agents': [find_ql.BasicAgent('find-basic-ql-agent-2', 2, batch=1)]
		},
		'find-fancy-ql': {
			'engine': 'find', 'args': [2],
			'agents': [find_ql.FancyAgent('find-fancy-ql-agent', 2, batch=100)]
		},
		'find-tg': {
			'engine': 'find', 'args': [2],
			'agents': [find_tg.Agent('find-tg-agent', 2)]
		},
		'fl-basic-rw': {
			'engine': 'frozenlake',
			'agents': [fl_ql.BasicAgent('fl-basic-rw-agent', gamma=0)]
		},
		'fl-basic-ql': {
			'engine': 'frozenlake',
			'agents': [fl_ql.BasicAgent('fl-basic-ql-agent', batch=1)]
		},
		'fl-fancy-ql': {
			'engine': 'frozenlake',
			'agents': [fl_ql.FancyAgent('fl-fancy-ql-agent', batch=100)]
		},
		'fl-tg': {
			'engine': 'frozenlake',
			'agents': [fl_tg.Agent('fl-tg-agent')]
		},
		'cp-basic-rw': {
			'engine': 'cartpole',
			'agents': [cp_ql.BasicAgent('cp-basic-rw-agent', gamma=0)]
		},
		'cp-basic-ql': {
			'engine': 'cartpole',
			'agents': [cp_ql.BasicAgent('cp-basic-ql-agent', batch=1)]
		},
		'cp-fancy-ql': {
			'engine': 'cartpole',
			'agents': [cp_ql.FancyAgent('cp-fancy-ql-agent', batch=100)]
		}
	}

	curriculum = {name: curriculum[name] for name in ('find-basic-rw', 'find-basic-ql')}
	School(curriculum).run(limit=5e5)
