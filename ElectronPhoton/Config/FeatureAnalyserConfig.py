import numpy as np
import math as m

############################################## FEATURE ANALYSER CONFIGURATION ##################################################

features = [
    #{'name': 'TotalChargeU', 'algorithmName': 'TotalCharge', 'bins': np.linspace(0, 100000, num = 30), 'cutDirection': 'left'},
    #{'name': 'TotalChargeV', 'algorithmName': 'TotalCharge', 'bins': np.linspace(0, 100000, num = 30), 'cutDirection': 'left'},
    #{'name': 'TotalChargeW', 'algorithmName': 'TotalCharge', 'bins': np.linspace(0, 100000, num = 50), 'cutDirection': 'left'},
    #{'name': 'TotalCharge3D', 'algorithmName': 'TotalCharge', 'bins': np.linspace(0, 100000, num = 30), 'cutDirection': 'left'},
    #{'name': 'LongitudinalSpanU', 'algorithmName': 'AngularSpan', 'bins': np.linspace(0, 400, num=50), 'cutDirection': 'left'},
    #{'name': 'LongitudinalSpanV', 'algorithmName': 'AngularSpan', 'bins': np.linspace(0, 600, num=50), 'cutDirection': 'left'},
    #{'name': 'LongitudinalSpanW', 'algorithmName': 'AngularSpan', 'bins': np.linspace(0, 600, num=50), 'cutDirection': 'left'},
    #{'name': 'LongitudinalSpan3D', 'algorithmName': 'AngularSpan', 'bins': np.linspace(0, 600, num=50), 'cutDirection': 'left'},
    #{'name': 'VertexSepU', 'algorithmName': 'VertexSeparation', 'bins': np.linspace(0, 10, num=30), 'cutDirection': 'left'},
    #{'name': 'VertexSepV', 'algorithmName': 'VertexSeparation', 'bins': np.linspace(0, 10, num=30), 'cutDirection': 'left'},
    #{'name': 'VertexSepW', 'algorithmName': 'VertexSeparation', 'bins': np.linspace(0, 10, num=30), 'cutDirection': 'left'},
    #{'name': 'VertexSep3D', 'algorithmName': 'VertexSeparation', 'bins': np.linspace(0, 10, num=30), 'cutDirection': 'left'},
    #{'name': 'TestDeDxU', 'algorithmName': 'TestDeDx1', 'bins': np.linspace(0, 500, num=50), 'cutDirection': 'left', 'cutPlot': 'simple'},
    #{'name': 'TestDeDxV', 'algorithmName': 'TestDeDx1', 'bins': np.linspace(0, 500, num=50), 'cutDirection': 'left', 'cutPlot': 'simple'},
    #{'name': 'TestDeDxW', 'algorithmName': 'TestDeDx1', 'bins': np.linspace(0, 100, num=50), 'cutDirection': 'left', 'cutPlot': 'simple'},
    #{'name': 'TestDeDx3D', 'algorithmName': 'TestDeDx1', 'bins': np.linspace(0, 500, num=50), 'cutDirection': 'left', 'cutPlot': 'simple'},
    #{'name': 'TestDeDx2U', 'algorithmName': 'TestDeDx2', 'bins': np.linspace(0, 1200, num=50), 'cutDirection': 'left', 'cutPlot': 'simple'},
    #{'name': 'TestDeDx2V', 'algorithmName': 'TestDeDx2', 'bins': np.linspace(0, 1200, num=50), 'cutDirection': 'left', 'cutPlot': 'simple'},
    #{'name': 'TestDeDx2W', 'algorithmName': 'TestDeDx2', 'bins': np.linspace(0, 1200, num=50), 'cutDirection': 'left', 'cutPlot': 'simple'},
    #{'name': 'TestDeDx2_3D', 'algorithmName': 'TestDeDx2', 'bins': np.linspace(0, 2000, num=50), 'cutDirection': 'left', 'cutPlot': 'simple'},
    #{'name': 'dedxU', 'algorithmName': 'NewInitialDeDx', 'bins': np.linspace(0, 2000, num=50), 'cutDirection': 'right'},
    #{'name': 'dedxV', 'algorithmName': 'NewInitialDeDx', 'bins': np.linspace(0, 2000, num=50), 'cutDirection': 'right'},
    #{'name': 'dedxW', 'algorithmName': 'NewInitialDeDx', 'bins': np.linspace(0, 2000, num=50), 'cutDirection': 'left', 'cutPlot': 'simple'},
    #{'name': 'BDTU', 'algorithmName': 'BDTCalculator', 'bins': np.linspace(-5, 10, num = 30), 'cutDirection': 'left'},
    #{'name': 'BDTV', 'algorithmName': 'BDTCalculator', 'bins': np.linspace(-5, 10, num = 30), 'cutDirection': 'left'},
    #{'name': 'BDTW', 'algorithmName': 'BDTCalculator', 'bins': np.linspace(-5, 10, num = 30), 'cutDirection': 'left'},
    #{'name': 'BDT3D', 'algorithmName': 'BDTCalculator', 'bins': np.linspace(-5, 10, num = 30), 'cutDirection': 'left'},
    #{'name': 'BDTMulti', 'algorithmName': 'BDTCalculator', 'bins': np.linspace(-5, 10, num = 30), 'cutDirection': 'left'},
    #{'name': 'Likelihood', 'algorithmName': 'LikelihoodCalculator', 'bins': np.linspace(0, 1, num = 200), 'cutDirection': 'right'},
    {'name': 'AngleFromX_3D', 'algorithmName': 'AnglesFromAxes', 'bins': np.linspace(0, m.pi, num = 50), 'cutDirection': 'left'},
    {'name': 'AngleFromY_3D', 'algorithmName': 'AnglesFromAxes', 'bins': np.linspace(0, m.pi, num = 50), 'cutDirection': 'left'},
    {'name': 'AngleFromZ_3D', 'algorithmName': 'AnglesFromAxes', 'bins': np.linspace(0, m.pi, num = 50), 'cutDirection': 'left'},
    {'name': 'mcpMomentum', 'algorithmName': 'GeneralInfo', 'bins': np.linspace(0, 0.5, num = 50), 'cutDirection': 'left', 'cutPlot': 'simple', 'cutFixed': 0.5},
    #{'name': 'nuMcpMomentum', 'algorithmName': 'GeneralInfo', 'bins': np.linspace(0, 2, num = 30), 'cutDirection': 'right'},
    #{'name': 'nHitsW', 'algorithmName': 'GeneralInfo', 'bins': np.linspace(0, 600, num=30), 'cutDirection': 'left'},
    #{'name': 'purityW', 'algorithmName': 'GeneralInfo', 'bins': np.linspace(0, 1, num=30), 'cutDirection': 'left'},
    #{'name': 'completenessW', 'algorithmName': 'GeneralInfo', 'bins': np.linspace(0, 1, num=30), 'cutDirection': 'left'},
]

featureHistogram = {
    "plot": True,
    "filters": (
        ("Photon", "abs(mcPdgCode)==22", "count", True),
        ("Electron", "abs(mcPdgCode)==11", "count", True),
        #("NCDelta Photon", "abs(mcPdgCode)==22 and dataName=='NCDelta'", "count", False),
        #("Pi0 Photon", "abs(mcPdgCode)==22 and dataName=='Pi0_0-1000'", "count", False),
        #("Electrons ncdelta+pi0", "abs(mcPdgCode)==11 and dataName in ('NCDelta', 'Pi0_0-1000')", "abs(mcPdgCode) in (11, 22)", True),
        #("Photons eminus", "abs(mcPdgCode)==22 and dataName == 'eminus_0-4000'", "abs(mcPdgCode) in (11, 22)", True),
        #("Nu_e primary e", "abs(mcPdgCode)==11 and abs(nuMcPdgCode)==12 and mcHierarchyTier==1", "abs(mcPdgCode) in (11, 22)", False),
        #("Nu_mu secondary e", "abs(mcPdgCode)==11 and abs(nuMcPdgCode)==14 and mcHierarchyTier==2", "abs(mcPdgCode) in (11, 22)", False),
        #("Electrons (Nu_e)", "abs(nuMcPdgCode)==12 and abs(mcPdgCode)==11", "abs(mcPdgCode)==11", False), 
        #("Electrons (Nu_e)", "abs(nuMcPdgCode)==12 and abs(mcPdgCode)==11", "abs(mcPdgCode)==11", False), 
        #("Electrons (Nu_mu)", "abs(nuMcPdgCode)==14 and abs(mcPdgCode)==11", "abs(mcPdgCode)==11", False),
        #("Photons, Nu_e primary", "abs(mcPdgCode)==22 and abs(nuMcPdgCode)==13 and mcHierarchyTier==1", "abs(mcPdgCode) in (11, 22)", False),
        #("Photons, Nu_e secondary", "abs(mcPdgCode)==22 and abs(nuMcPdgCode)==12 and mcHierarchyTier==2", "abs(mcPdgCode) in (11, 22)", False),
        #("Photons, Nu_e tertiary or higher", "abs(mcPdgCode)==22 and abs(nuMcPdgCode)==12 and mcHierarchyTier>2", "abs(mcPdgCode) in (11, 22)", False),
        #("Photons, Nu_mu primary", "abs(mcPdgCode)==22 and abs(nuMcPdgCode)==14 and mcHierarchyTier==1", "abs(mcPdgCode) in (11, 22)", False),
        #("Nu_mu secondary photon", "abs(mcPdgCode)==22 and abs(nuMcPdgCode)==14 and mcHierarchyTier==2", "abs(mcPdgCode) in (11, 22)", False),
        #("Photons, Nu_mu tertiary or higher", "abs(mcPdgCode)==22 and abs(nuMcPdgCode)==14 and mcHierarchyTier>2", "abs(mcPdgCode) in (11, 22)", False),
    )   
}

purityEfficiency = {
    "plot": True,
    "nTestCuts": 1000
}