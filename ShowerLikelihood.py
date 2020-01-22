import DataSampler as ds
import LikelihoodCalculator as lc
import LikelihoodAnalyser as la
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
            "name": "Electrons",
            "query": "abs(mcPdgCode)==11"
        },
        "showPurity": False
    },
    {
        "dependence": "nHitsU+nHitsV+nHitsW",
        'bins': np.linspace(0, 800, num=40),
        "pfoClass": "shower",
        "filter": {
            "name": "Photons",
            "query": "abs(mcPdgCode)==22"
        },
        "showPurity": False
    },
    {
        "dependence": "mcpMomentum",
        'bins': np.linspace(0, 1, num=40),
        "pfoClass": "shower",
        "filter": {
            "name": "Electrons",
            "query": "abs(mcPdgCode)==11"
        },
        "showPurity": False
    },
    {
        "dependence": "mcpMomentum",
        'bins': np.linspace(0, 1, num=80),
        "pfoClass": "shower",
        "filter": {
            "name": "Photons",
            "query": "abs(mcPdgCode)==22"
        },
        "showPurity": False
    },
)


if __name__ == "__main__":
    
    ds.GetPerfPfoData(viewsUsed=lc.viewsUsed)
    cutoffValues, cutoffResults = la.GetBestPurityEfficiency(ds.dfPerfPfoData['track']['union'], ds.dfPerfPfoData['shower']['union'], ('track', 'shower'), {'name': 'Likelihood', 'bins': (0, 1), 'cutDirection': 'right'}, purityEfficiencyVsCutoffGraph['nTestCuts'])

    for histogram in likelihoodHistograms:
        histogram['name'] = 'Likelihood'
        la.PlotVariableHistogram(ds.dfPerfPfoData['all']['union'], ('track', 'shower'), {'name': 'Likelihood', 'cutDirection': 'right'}, histogram, cutoffResults[4])

    la.PlotPurityEfficiencyVsCutoff("Likelihood", ("track", "shower"), cutoffValues, cutoffResults)
    
    for graph in purityEfficiencyBinnedGraphs:
        graph["cutoff"] = cutoffResults[4]
        la.PlotPurityEfficiencyVsVariable(ds.dfPerfPfoData["track"]['union'], ds.dfPerfPfoData["shower"]["union"], ("track", "shower"), graph)