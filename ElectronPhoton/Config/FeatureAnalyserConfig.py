import numpy as np
import math as m

############################################## FEATURE ANALYSER CONFIGURATION ##################################################

features = [
    #{'name': 'InitialdEdxU', 'algorithmName': 'InitialdEdx', 'bins': np.linspace(0, 20000, num = 50), 'cutDirection': 'left'},
    #{'name': 'InitialdEdxV', 'algorithmName': 'InitialdEdx', 'bins': np.linspace(0, 20000, num = 50), 'cutDirection': 'left'},
    #{'name': 'InitialdEdxW', 'algorithmName': 'InitialdEdx', 'bins': np.linspace(0, 20000, num = 50), 'cutDirection': 'left'},
    #{'name': 'InitialdEdx3D', 'algorithmName': 'InitialdEdx', 'bins': np.linspace(0, 20000, num = 50), 'cutDirection': 'left'},
    #{'name': 'LongitudinalSpanU', 'algorithmName': 'AngularSpan', 'bins': np.linspace(0, 400, num=50), 'cutDirection': 'left'},
    #{'name': 'LongitudinalSpanV', 'algorithmName': 'AngularSpan', 'bins': np.linspace(0, 600, num=50), 'cutDirection': 'left'},
    #{'name': 'LongitudinalSpanW', 'algorithmName': 'AngularSpan', 'bins': np.linspace(0, 600, num=50), 'cutDirection': 'left'},
    #{'name': 'LongitudinalSpan3D', 'algorithmName': 'AngularSpan', 'bins': np.linspace(0, 600, num=50), 'cutDirection': 'left'},
    {'name': 'VertexSepU', 'algorithmName': 'VertexSeparation', 'bins': np.linspace(0, 10, num=50), 'cutDirection': 'left'},
    {'name': 'VertexSepV', 'algorithmName': 'VertexSeparation', 'bins': np.linspace(0, 10, num=50), 'cutDirection': 'left'},
    {'name': 'VertexSepW', 'algorithmName': 'VertexSeparation', 'bins': np.linspace(0, 10, num=50), 'cutDirection': 'left'},
    {'name': 'VertexSep3D', 'algorithmName': 'VertexSeparation', 'bins': np.linspace(0, 10, num=50), 'cutDirection': 'left'},

    #{'name': 'BDTU', 'algorithmName': 'DecisionTreeCalculator', 'bins': np.linspace(-10, 15, num = 200), 'cutDirection': 'left'},
    #{'name': 'BDTV', 'algorithmName': 'DecisionTreeCalculator', 'bins': np.linspace(-10, 15, num = 200), 'cutDirection': 'left'},
    #{'name': 'BDTW', 'algorithmName': 'DecisionTreeCalculator', 'bins': np.linspace(-10, 15, num = 200), 'cutDirection': 'left'},
    #{'name': 'BDT3D', 'algorithmName': 'DecisionTreeCalculator', 'bins': np.linspace(-10, 15, num = 200), 'cutDirection': 'left'},
    #{'name': 'BDTMulti', 'algorithmName': 'DecisionTreeCalculator', 'bins': np.linspace(-10, 15, num = 200), 'cutDirection': 'left'},
    #{'name': 'Likelihood', 'algorithmName': 'LikelihoodCalculator', 'bins': np.linspace(0, 1, num = 200), 'cutDirection': 'right'},
    #{'name': 'mcpMomentum', 'algorithmName': 'GeneralInfo', 'bins': np.linspace(0, 0.3, num = 100), 'cutDirection': 'left'},
]

featureHistogram = {
    "plot": True,
    "filters": (
        ("Electrons", "abs(mcPdgCode)==11", "", True), 
        ("Photons", "abs(mcPdgCode)==22", "", True),
    )   
}

purityEfficiency = {
    "plot": True,
    "nTestCuts": 1000
}