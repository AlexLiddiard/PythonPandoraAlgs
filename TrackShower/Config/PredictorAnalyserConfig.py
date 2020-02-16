import numpy as np
from UpRootFileReader import MicroBooneGeo

############################################## PREDICTOR ANALYSER CONFIGURATION ##################################################

#predictor = {"name": "Likelihood", "algorithmName": "LikelihoodCalculator", "range": (0, 1), 'cutDirection': 'right', 'plotCut': 'fancy'}
predictor = {"name": "BDTMulti", "algorithmName": "BDTCalculator", "range": (-10, 10), 'cutDirection': 'left', 'plotCut': 'fancy'}

predictorHistograms = [
    {
        'filters': (
            ('Showers', 'isShower==1', 'isShower in (0, 1)', True),
            ('Tracks', 'isShower==0', 'isShower in (0, 1)', True)
        ),
        'bins': np.linspace(*predictor["range"], num=50),
        'yAxis': 'log',
        'cutoff': 'shower'
    },
    {
        'filters': (
            ('Showers', 'isShower==1', 'isShower in (0, 1)', True),
            ('Electrons + Positrons', 'abs(mcPdgCode)==11', 'isShower in (0, 1)', False),
            ('Photons', 'abs(mcPdgCode)==22', 'isShower in (0, 1)', False)
        ),
        'bins': np.linspace(*predictor["range"], num=25),
        'yAxis': 'log',
        'cutoff': 'shower'
    },
    {
        'filters': (
            ('Tracks', 'isShower in (0, 1)', '', True),
            ('Protons', 'abs(mcPdgCode)==2212', 'isShower in (0, 1)', False),
            ('Muons', 'abs(mcPdgCode)==13', 'isShower in (0, 1)', False),
            ('Charged Pions', 'abs(mcPdgCode)==211', 'isShower in (0, 1)', False)
        ),
        'bins': np.linspace(*predictor["range"], num=25),
        'yAxis': 'log',
        'cutoff': 'track'
    },
]

purityEfficiencyVsCutoffGraph = {'nTestCuts': 1001}
purityEfficiencyBinnedGraphs = [
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
    #{
    #    "dependence": "minCoordY3D",
    #    'bins': np.linspace(MicroBooneGeo.RangeY[0], MicroBooneGeo.RangeY[1], num=40),
    #    "pfoClass": "both"
    #},
    #{
    #    "dependence": "maxCoordY3D",
    #    'bins': np.linspace(MicroBooneGeo.RangeY[0], MicroBooneGeo.RangeY[1], num=40),
    #    "pfoClass": "both"
    #},
    #{
    #    "dependence": "minCoordZ3D",
    #    'bins': np.linspace(MicroBooneGeo.RangeZ[0], MicroBooneGeo.RangeZ[1], num=40),
    #    "pfoClass": "both"
    #},
    #{
    #    "dependence": "maxCoordZ3D",
    #    'bins': np.linspace(MicroBooneGeo.RangeZ[0], MicroBooneGeo.RangeZ[1], num=40),
    #    "pfoClass": "both"
    #},
    #{
    #    "dependence": "nHitsU+nHitsV+nHitsW",
    #    'bins': np.linspace(0, 1400, num=40),
    #    "pfoClass": "both"
    #},
    {
        "dependence": "mcpMomentum",
        'bins': np.linspace(0, 1.5, num=40),
        "pfoClass": "both"
    },
    {
        "dependence": "purityW",
        'bins': np.linspace(0, 1, num=40),
        "pfoClass": "both"
    },
    {
        "dependence": "completenessW", 
        'bins': np.linspace(0, 1, num=40), 
        "pfoClass": "both"
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