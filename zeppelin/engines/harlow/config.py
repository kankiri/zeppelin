observation_shapes = {
	'position': {
		'type': int,
		'shape': (2,)
	},
	'reward': {
		'type': float,
		'optional': True
	},
	'done': {
		'type': bool,
		'optional': True
	}
}

action_shapes = {
	'action': {
		'type': int,
		'min': 0,
		'max': 1
	}
}