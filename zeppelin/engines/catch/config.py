observation_shapes = {
	'position': {
		'type': float
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
		'min': 0
	}
}