import numpy as np
import math as m

############################################## FEATURE ANALYSER CONFIGURATION ##################################################

features = [
    #{'name': 'TotalChargeU', 'algorithmName': 'TotalCharge', 'bins': np.linspace(0, 100000, num = 30), 'cutDirection': 'left'},
    #{'name': 'TotalChargeV', 'algorithmName': 'TotalCharge', 'bins': np.linspace(0, 100000, num = 30), 'cutDirection': 'left'},
    #{'name': 'TotalChargeW', 'algorithmName': 'TotalCharge', 'bins': np.linspace(0, 100000, num = 30), 'cutDirection': 'left'},
    #{'name': 'TotalCharge3D', 'algorithmName': 'TotalCharge', 'bins': np.linspace(0, 100000, num = 30), 'cutDirection': 'left'},
    #{'name': 'LongitudinalSpanU', 'algorithmName': 'AngularSpan', 'bins': np.linspace(0, 400, num=50), 'cutDirection': 'left'},
    #{'name': 'LongitudinalSpanV', 'algorithmName': 'AngularSpan', 'bins': np.linspace(0, 600, num=50), 'cutDirection': 'left'},
    #{'name': 'LongitudinalSpanW', 'algorithmName': 'AngularSpan', 'bins': np.linspace(0, 600, num=50), 'cutDirection': 'left'},
    #{'name': 'LongitudinalSpan3D', 'algorithmName': 'AngularSpan', 'bins': np.linspace(0, 600, num=50), 'cutDirection': 'left'},
    #{'name': 'VertexSepU', 'algorithmName': 'VertexSeparation', 'bins': np.linspace(0, 10, num=30), 'cutDirection': 'left'},
    #{'name': 'VertexSepV', 'algorithmName': 'VertexSeparation', 'bins': np.linspace(0, 10, num=30), 'cutDirection': 'left'},
    #{'name': 'VertexSepW', 'algorithmName': 'VertexSeparation', 'bins': np.linspace(0, 10, num=30), 'cutDirection': 'left'},
    #{'name': 'VertexSep3D', 'algorithmName': 'VertexSeparation', 'bins': np.linspace(0, 10, num=30), 'cutDirection': 'left'},
    #{'name': 'TestDeDxU', 'algorithmName': 'TestDeDx1', 'bins': np.linspace(0, 1, num=50), 'cutDirection': 'left'},
    #{'name': 'TestDeDxV', 'algorithmName': 'TestDeDx1', 'bins': np.linspace(0, 1, num=50), 'cutDirection': 'left'},
    #{'name': 'TestDeDxW', 'algorithmName': 'TestDeDx1', 'bins': np.linspace(0, 1, num=50), 'cutDirection': 'left'},
    #{'name': 'TestDeDx3D', 'algorithmName': 'TestDeDx1', 'bins': np.linspace(0, 1, num=50), 'cutDirection': 'left'},
    #{'name': 'TestDeDx2U', 'algorithmName': 'TestDeDx2', 'bins': np.linspace(0, 1200, num=50), 'cutDirection': 'left'},
    #{'name': 'TestDeDx2V', 'algorithmName': 'TestDeDx2', 'bins': np.linspace(0, 1200, num=50), 'cutDirection': 'left'},
    #{'name': 'TestDeDx2W', 'algorithmName': 'TestDeDx2', 'bins': np.linspace(0, 1200, num=50), 'cutDirection': 'left'},
    #{'name': 'dedxU', 'algorithmName': 'NewInitialDeDx', 'bins': np.linspace(0, 2000, num=30), 'cutDirection': 'left'},
    #{'name': 'dedxV', 'algorithmName': 'NewInitialDeDx', 'bins': np.linspace(0, 2000, num=30), 'cutDirection': 'left'},
    #{'name': 'dedxW', 'algorithmName': 'NewInitialDeDx', 'bins': np.linspace(0, 2000, num=30), 'cutDirection': 'left'},
    {'name': 'BDTU', 'algorithmName': 'BDTCalculator', 'bins': np.linspace(-5, 10, num = 30), 'cutDirection': 'left'},
    {'name': 'BDTV', 'algorithmName': 'BDTCalculator', 'bins': np.linspace(-5, 10, num = 30), 'cutDirection': 'left'},
    {'name': 'BDTW', 'algorithmName': 'BDTCalculator', 'bins': np.linspace(-5, 10, num = 30), 'cutDirection': 'left'},
    {'name': 'BDT3D', 'algorithmName': 'BDTCalculator', 'bins': np.linspace(-5, 10, num = 30), 'cutDirection': 'left'},
    {'name': 'BDTMulti', 'algorithmName': 'BDTCalculator', 'bins': np.linspace(-5, 10, num = 30), 'cutDirection': 'left'},
    #{'name': 'Likelihood', 'algorithmName': 'LikelihoodCalculator', 'bins': np.linspace(0, 1, num = 200), 'cutDirection': 'right'},
    {'name': 'mcpMomentum', 'algorithmName': 'GeneralInfo', 'bins': np.linspace(0, 3, num = 30), 'cutDirection': 'right'},
    {'name': 'nuMcpMomentum', 'algorithmName': 'GeneralInfo', 'bins': np.linspace(0, 2, num = 30), 'cutDirection': 'right'},
    {'name': 'nHitsW', 'algorithmName': 'GeneralInfo', 'bins': np.linspace(0, 600, num=30), 'cutDirection': 'left'},
    {'name': 'purityW', 'algorithmName': 'GeneralInfo', 'bins': np.linspace(0, 1, num=30), 'cutDirection': 'left'},
    {'name': 'completenessW', 'algorithmName': 'GeneralInfo', 'bins': np.linspace(0, 1, num=30), 'cutDirection': 'left'},
]

featureHistogram = {
    "plot": True,
    "filters": (
        ("e", "abs(mcPdgCode)==11", "abs(mcPdgCode) in (11, 22)", True),
        ("Nu_e primary e", "abs(mcPdgCode)==11 and abs(nuMcPdgCode)==12 and mcHierarchyTier==1", "abs(mcPdgCode) in (11, 22)", False),
        ("Nu_mu secondary e", "abs(mcPdgCode)==11 and abs(nuMcPdgCode)==14 and mcHierarchyTier==2", "abs(mcPdgCode) in (11, 22)", False),
        #("Electrons (Nu_e)", "abs(nuMcPdgCode)==12 and abs(mcPdgCode)==11", "abs(mcPdgCode)==11", False), 
        #("Electrons (Nu_e)", "abs(nuMcPdgCode)==12 and abs(mcPdgCode)==11", "abs(mcPdgCode)==11", False), 
        #("Electrons (Nu_mu)", "abs(nuMcPdgCode)==14 and abs(mcPdgCode)==11", "abs(mcPdgCode)==11", False),
        ("Photon", "abs(mcPdgCode)==22", "", True),
        #("Photons, Nu_e primary", "abs(mcPdgCode)==22 and abs(nuMcPdgCode)==13 and mcHierarchyTier==1", "abs(mcPdgCode) in (11, 22)", False),
        #("Photons, Nu_e secondary", "abs(mcPdgCode)==22 and abs(nuMcPdgCode)==12 and mcHierarchyTier==2", "abs(mcPdgCode) in (11, 22)", False),
        #("Photons, Nu_e tertiary or higher", "abs(mcPdgCode)==22 and abs(nuMcPdgCode)==12 and mcHierarchyTier>2", "abs(mcPdgCode) in (11, 22)", False),
        #("Photons, Nu_mu primary", "abs(mcPdgCode)==22 and abs(nuMcPdgCode)==14 and mcHierarchyTier==1", "abs(mcPdgCode) in (11, 22)", False),
        ("Nu_mu secondary photon", "abs(mcPdgCode)==22 and abs(nuMcPdgCode)==14 and mcHierarchyTier==2", "abs(mcPdgCode) in (11, 22)", False),
        #("Photons, Nu_mu tertiary or higher", "abs(mcPdgCode)==22 and abs(nuMcPdgCode)==14 and mcHierarchyTier>2", "abs(mcPdgCode) in (11, 22)", False),
    )   
}

purityEfficiency = {
    "plot": True,
    "nTestCuts": 1000
}