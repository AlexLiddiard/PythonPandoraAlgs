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
    #{'name': 'VertexSepU', 'algorithmName': 'VertexSeparation', 'bins': np.linspace(0, 10, num=50), 'cutDirection': 'left'},
    #{'name': 'VertexSepV', 'algorithmName': 'VertexSeparation', 'bins': np.linspace(0, 10, num=50), 'cutDirection': 'left'},
    #{'name': 'VertexSepW', 'algorithmName': 'VertexSeparation', 'bins': np.linspace(0, 10, num=50), 'cutDirection': 'left'},
    #{'name': 'VertexSep3D', 'algorithmName': 'VertexSeparation', 'bins': np.linspace(0, 10, num=50), 'cutDirection': 'left'},
    #{'name': 'TestDeDxU', 'algorithmName': 'TestDeDx1', 'bins': np.linspace(0, 1, num=50), 'cutDirection': 'left'},
    #{'name': 'TestDeDxV', 'algorithmName': 'TestDeDx1', 'bins': np.linspace(0, 1, num=50), 'cutDirection': 'left'},
    #{'name': 'TestDeDxW', 'algorithmName': 'TestDeDx1', 'bins': np.linspace(0, 1, num=50), 'cutDirection': 'left'},
    #{'name': 'TestDeDx3D', 'algorithmName': 'TestDeDx1', 'bins': np.linspace(0, 1, num=50), 'cutDirection': 'left'},
    #{'name': 'TestDeDx2U', 'algorithmName': 'TestDeDx2', 'bins': np.linspace(0, 1200, num=50), 'cutDirection': 'left'},
    #{'name': 'TestDeDx2V', 'algorithmName': 'TestDeDx2', 'bins': np.linspace(0, 1200, num=50), 'cutDirection': 'left'},
    #{'name': 'TestDeDx2W', 'algorithmName': 'TestDeDx2', 'bins': np.linspace(0, 1200, num=50), 'cutDirection': 'left'},
    {'name': 'dedxU', 'algorithmName': 'NewInitialDeDx', 'bins': np.linspace(0, 2000, num=50), 'cutDirection': 'left'},
    {'name': 'dedxV', 'algorithmName': 'NewInitialDeDx', 'bins': np.linspace(0, 2000, num=50), 'cutDirection': 'left'},
    {'name': 'dedxW', 'algorithmName': 'NewInitialDeDx', 'bins': np.linspace(0, 2000, num=50), 'cutDirection': 'left'},
    #{'name': 'BDTU', 'algorithmName': 'DecisionTreeCalculator', 'bins': np.linspace(-10, 15, num = 200), 'cutDirection': 'left'},
    #{'name': 'BDTV', 'algorithmName': 'DecisionTreeCalculator', 'bins': np.linspace(-10, 15, num = 200), 'cutDirection': 'left'},
    #{'name': 'BDTW', 'algorithmName': 'DecisionTreeCalculator', 'bins': np.linspace(-10, 15, num = 200), 'cutDirection': 'left'},
    #{'name': 'BDT3D', 'algorithmName': 'DecisionTreeCalculator', 'bins': np.linspace(-10, 15, num = 200), 'cutDirection': 'left'},
    #{'name': 'BDTMulti', 'algorithmName': 'DecisionTreeCalculator', 'bins': np.linspace(-10, 15, num = 200), 'cutDirection': 'left'},
    #{'name': 'Likelihood', 'algorithmName': 'LikelihoodCalculator', 'bins': np.linspace(0, 1, num = 200), 'cutDirection': 'right'},
    #{'name': 'mcpMomentum', 'algorithmName': 'GeneralInfo', 'bins': np.linspace(0, 0.6, num = 50), 'cutDirection': 'left'},
    #{'name': 'nHitsW', 'algorithmName': 'GeneralInfo', 'bins': np.linspace(0, 600, num=50), 'cutDirection': 'left'},
    #{'name': 'purityW', 'algorithmName': 'GeneralInfo', 'bins': np.linspace(0, 1, num=50), 'cutDirection': 'left'},
    #{'name': 'completenessW', 'algorithmName': 'GeneralInfo', 'bins': np.linspace(0, 1, num=50), 'cutDirection': 'left'},
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