import numpy as np
from UpRootFileReader import MicroBooneGeo

############################################## PREDICTOR ANALYSER CONFIGURATION ##################################################

predictors = [
    {"name": "Likelihood", "algorithmName": "LikelihoodCalculator", "bins": np.linspace(0, 1, 50), 'cutDirection': 'right', 'cutPlot': 'simple'},
    {"name": "BDTMulti", "algorithmName": "BDTCalculator", "bins": np.linspace(-10, 11, 50), 'cutDirection': 'left', 'cutPlot': 'simple'}
]

predictorHistograms = [
    {
        'filters': (
            ('Showers', 'isShower==1', 'count', True),
            ('Tracks', 'isShower==0', 'count', True)
        ),
        'yAxis': 'log',
        'cutoff': 'shower'
    },
    {
        'filters': (
            ('Showers', 'isShower==1', 'count', True),
            ('Electrons + Positrons', 'abs(mcPdgCode)==11', 'count', False),
            ('Photons', 'abs(mcPdgCode)==22', 'count', False)
        ),
        'yAxis': 'log',
        'cutoff': 'shower'
    },
    {
        'filters': (
            ('Tracks', 'isShower==0', 'count', True),
            ('Protons', 'abs(mcPdgCode)==2212', 'count', False),
            ('Muons', 'abs(mcPdgCode)==13', 'count', False),
            ('Charged Pions', 'abs(mcPdgCode)==211', 'count', False),
        ),
        'yAxis': 'log',
        'cutoff': 'track'
    },
]

purityEfficiencyVsCutoffGraph = {'nTestCuts': 1001}
purityEfficiencyBinnedGraphs = [
    #{
    #    "dependence": "nHitsW", 
    #    'bins': np.linspace(0, 400, num=40), 
    #    "pfoClass": "track", 
    #    "filter": {
    #        "name": "NCDIS muon",
    #        "query": "abs(mcNuPdgCode) == 14 and abs(mcParentPdgCode) == 14 and abs(mcPdgCode) == 2212 and mcHierarchyTier == 1 and mcNuanceCode == 1002"
    #    },
    #    "showPurity": False
    #},
    {
        "dependence": "minCoordX3D",
        'bins': np.linspace(MicroBooneGeo.RangeX[0], MicroBooneGeo.RangeX[1], num=40),
        "pfoClass": "both"
    },
    {
        "dependence": "maxCoordX3D",
        'bins': np.linspace(MicroBooneGeo.RangeX[0], MicroBooneGeo.RangeX[1], num=40),
        "pfoClass": "both"
    },
    {
        "dependence": "minCoordY3D",
        'bins': np.linspace(MicroBooneGeo.RangeY[0], MicroBooneGeo.RangeY[1], num=40),
        "pfoClass": "both"
    },
    {
        "dependence": "maxCoordY3D",
        'bins': np.linspace(MicroBooneGeo.RangeY[0], MicroBooneGeo.RangeY[1], num=40),
        "pfoClass": "both"
    },
    {
        "dependence": "minCoordZ3D",
        'bins': np.linspace(MicroBooneGeo.RangeZ[0], MicroBooneGeo.RangeZ[1], num=40),
        "pfoClass": "both"
    },
    {
        "dependence": "maxCoordZ3D",
        'bins': np.linspace(MicroBooneGeo.RangeZ[0], MicroBooneGeo.RangeZ[1], num=40),
        "pfoClass": "both"
    },
    {
        "dependence": "nHitsU+nHitsV+nHitsW",
        'bins': np.linspace(0, 1400, num=40),
        "pfoClass": "both",
        "showPurity": False
    },
    {
        "dependence": "mcpMomentum",
        'bins': np.linspace(0, 1.5, num=40),
        "pfoClass": "both",
        "showPurity": False
    },
    {
        "dependence": "nHitsW", 
        'bins': np.linspace(0, 400, num=40), 
        "pfoClass": "both",
        "showPurity": False
    },
    {
        "dependence": "purityW",
        'bins': np.linspace(0, 1, num=40),
        "pfoClass": "both",
        "showPurity": False
    },
    {
        "dependence": "completenessW", 
        'bins': np.linspace(0, 1, num=40), 
        "pfoClass": "both",
        "showPurity": False
    },
    {
        "dependence": "nHitsU+nHitsV+nHitsW", 
        'bins': np.linspace(0, 400, num=40), 
        "pfoClass": "shower", 
        "filter": {
            "name": "electron",
            "query": "abs(mcPdgCode)==11"
        },
        "showPurity": False
    },
    {
        "dependence": "nHitsU+nHitsV+nHitsW",
        'bins': np.linspace(0, 800, num=40),
        "pfoClass": "shower",
        "filter": {
            "name": "photon",
            "query": "abs(mcPdgCode)==22"
        },
        "showPurity": False
    },
    {
        "dependence": "mcpMomentum",
        'bins': np.linspace(0, 1, num=40),
        "pfoClass": "shower",
        "filter": {
            "name": "electron",
            "query": "abs(mcPdgCode)==11"
        },
        "showPurity": False
    },
    {
        "dependence": "mcpMomentum",
        'bins': np.linspace(0, 1, num=80),
        "pfoClass": "shower",
        "filter": {
            "name": "photon",
            "query": "abs(mcPdgCode)==22"
        },
        "showPurity": False
    },
    {
        "dependence": "mcpMomentum",
        'bins': np.linspace(0, 1, num=40),
        "pfoClass": "both",
        "filter": {
            "name": "CCQE",
            "query": "mcNuanceCode==1001"
        },
        "showPurity": True
    },
    {
        "dependence": "mcpMomentum",
        'bins': np.linspace(0, 1, num=80),
        "pfoClass": "both",
        "filter": {
            "name": "NCQE",
            "query": "mcNuanceCode==1002"
        },
        "showPurity": True
    },
    {
        "dependence": "mcpMomentum",
        'bins': np.linspace(0, 0.6, num=40),
        "pfoClass": "both",
        "filter": {
            "name": "CCRES",
            "query": "mcNuanceCode==1003 or mcNuanceCode==1004 or mcNuanceCode==1005"
        },
        "showPurity": True
    },
    {
        "dependence": "mcpMomentum",
        'bins': np.linspace(0, 1, num=80),
        "pfoClass": "both",
        "filter": {
            "name": "NCRES",
            "query": "mcNuanceCode==1006 or mcNuanceCode==1007 or mcNuanceCode==1008 or mcNuanceCode==1009"
        },
        "showPurity": True
    },
    {
        "dependence": "mcpMomentum",
        'bins': np.linspace(0, 1, num=80),
        "pfoClass": "both",
        "filter": {
            "name": "CCDIS",
            "query": "mcNuanceCode==1091"
        },
        "showPurity": True
    },
    {
        "dependence": "mcpMomentum",
        'bins': np.linspace(0, 1, num=80),
        "pfoClass": "both",
        "filter": {
            "name": "NCDIS",
            "query": "mcNuanceCode==1092"
        },
        "showPurity": True
    },
    {
        "dependence": "mcpMomentum",
        'bins': np.linspace(0, 1, num=80),
        "pfoClass": "both",
        "filter": {
            "name": "NCCOH",
            "query": "mcNuanceCode==1096"
        },
        "showPurity": True
    },
    {
        "dependence": "mcpMomentum",
        'bins': np.linspace(0, 1, num=80),
        "pfoClass": "both",
        "filter": {
            "name": "CCCOH",
            "query": "mcNuanceCode==1097"
        },
        "showPurity": True
    },
]