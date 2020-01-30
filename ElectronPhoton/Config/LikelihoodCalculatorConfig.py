import numpy as np

############################################## LIKELIHOOD CONFIGURATION ##################################################

features = [
    {'name': 'InitialdEdxU', 'algorithmName': 'InitialdEdx', 'bins': np.linspace(0, 5, num = 200), 'cutDirection': 'left'},
    {'name': 'InitialdEdxV', 'algorithmName': 'InitialdEdx', 'bins': np.linspace(0, 5, num = 200), 'cutDirection': 'left'},
    {'name': 'InitialdEdxW', 'algorithmName': 'InitialdEdx', 'bins': np.linspace(0, 5, num = 200), 'cutDirection': 'left'},
    {'name': 'InitialdEdx3D', 'algorithmName': 'InitialdEdx', 'bins': np.linspace(0, 5, num = 200), 'cutDirection': 'left'},
]