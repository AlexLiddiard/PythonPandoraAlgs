import numpy as np

likelihoodHistograms = (
    {
        'filters': (
            ('Showers', 'isShower==1', '', True),
            ('Tracks', 'isShower==0', '', True)
        ),
        'bins': np.linspace(0, 1, num=25),
        'yAxis': 'log',
        'cutoff': 'shower'
    },
    {
        'filters': (
            ('Showers', 'isShower==1', '', True),
            ('Electrons + Positrons', 'abs(mcPdgCode)==11', 'isShower==1', False),
            ('Photons', 'abs(mcPdgCode)==22', 'isShower==1', False)
        ),
        'bins': np.linspace(0, 1, num=25),
        'yAxis': 'log',
        'cutoff': 'shower'
    },
    {
        'filters': (
            ('Tracks', 'isShower==0', '', True),
            ('Protons', 'abs(mcPdgCode)==2212', 'isShower==0', False),
            ('Muons', 'abs(mcPdgCode)==13', 'isShower==0', False),
            ('Charged Pions', 'abs(mcPdgCode)==211', 'isShower==0', False)
        ),
        'bins': np.linspace(0, 1, num=25),
        'yAxis': 'log',
        'cutoff': 'track'
    },
)

purityEfficiencyVsCutoffGraph = {'nTestCuts': 1001}
purityEfficiencyBinnedGraphs = (
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
)