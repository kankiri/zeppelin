observation_shapes = {
	'position': {
		'type': int,
		'min': 0,
		'max': 15
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
		'max': 4
	}
}