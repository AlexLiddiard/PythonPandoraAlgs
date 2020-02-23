import numpy as np

############################################## PREDICTOR ANALYSER CONFIGURATION ##################################################

predictors = [
    {"name": "Likelihood", "algorithmName": "LikelihoodCalculator", 'bins': np.linspace(0, 1, num=50), 'cutDirection': 'right', 'cutPlot': 'simple'},
    #{"name": "BDTW", "algorithmName": "BDTCalculator", 'bins': np.linspace(-7, 7, num=50), 'cutDirection': 'left', 'cutPlot': 'simple'},
    #{"name": "BDTMulti", "algorithmName": "BDTCalculator", 'bins': np.linspace(-10, 10, num=50), 'cutDirection': 'left', 'cutPlot': 'simple'},
    {"name": "BDTAll", "algorithmName": "BDTCalculator", 'bins': np.linspace(-7, 7, num=50), 'cutDirection': 'left', 'cutPlot': 'simple'},
    {"name": "MVCNN", "algorithmName": "CNN", 'bins': np.linspace(0, 1, num = 3), 'cutDirection': 'right', 'cutPlot':'simple', 'cutFixed': 0.5},
]

predictorHistograms = [
    {
        'filters': (
            ('Electrons', 'abs(mcPdgCode)==11', 'count', True),
            ('Photons', 'abs(mcPdgCode)==22', 'count', True)
        ),
        'yAxis': 'log',
        'cutoff': 'electron'
    },
]

purityEfficiencyVsCutoffGraph = {'nTestCuts': 1001}
purityEfficiencyBinnedGraphs = [
    {
        "dependence":
        "nHitsW",
        'bins': np.linspace(0, 350, num=40),
        "pfoClass": "both",
        "showPurity": False
    },
    {
        "dependence": "mcpMomentum",
        'bins': np.linspace(0, 0.5, num=40),
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
        "dependence": "mcpMomentum",
        'bins': np.linspace(0, 0.5, num=80),
        "pfoClass": "both",
        "filter": {
            "name": "CCQE",
            "query": "mcNuanceCode==1001"
        },
        "showPurity": True
    },
    {
        "dependence": "mcpMomentum",
        'bins': np.linspace(0, 0.5, num=80),
        "pfoClass": "both",
        "filter": {
            "name": "NCQE",
            "query": "mcNuanceCode==1002"
        },
        "showPurity": True
    },
    {
        "dependence": "mcpMomentum",
        'bins': np.linspace(0, 0.5, num=80),
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
