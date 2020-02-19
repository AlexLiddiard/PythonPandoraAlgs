import numpy as np

############################################## PREDICTOR ANALYSER CONFIGURATION ##################################################

#predictor = {"name": "Likelihood", "algorithmName": "LikelihoodCalculator", "range": (0, 1), 'cutDirection': 'right'}
predictor = {"name": "BDTMulti", "algorithmName": "BDTCalculator", "range": (-10, 10), 'cutDirection': 'left', 'cutPlot': 'simple'}

predictorHistograms = [
    {
        'filters': (
            ('Electrons', 'abs(mcPdgCode)==11', 'count', True),
            ('Photons', 'abs(mcPdgCode)==22', 'count', True)
        ),
        'bins': np.linspace(*predictor["range"], num=25),
        'yAxis': 'log',
        'cutoff': 'electron'
    },
]

purityEfficiencyVsCutoffGraph = {'nTestCuts': 1001}
purityEfficiencyBinnedGraphs = [
    {
        "dependence":
        "nHitsU+nHitsV+nHitsW",
        'bins': np.linspace(60, 1400, num=40),
        "pfoClass": "both"
    },
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
        "dependence": "mcpMomentum",
        'bins': np.linspace(0, 1, num=80),
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
        'bins': np.linspace(0, 1, num=80),
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