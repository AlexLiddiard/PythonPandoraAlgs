import FeatureAnalyser as fa
import DataSampler as ds
import numpy as np

features = (
    {'name': 'InitialdEdxU', 'algorithmName': 'InitialdEdx', 'bins': np.linspace(0, 20000, num=100), 'cutDirection': 'left'},
    {'name': 'InitialdEdxV', 'algorithmName': 'InitialdEdx', 'bins': np.linspace(0, 20000, num=100), 'cutDirection': 'left'},
    {'name': 'InitialdEdxW', 'algorithmName': 'InitialdEdx', 'bins': np.linspace(0, 20000, num=100), 'cutDirection': 'left'},
    {'name': 'InitialdEdx3D', 'algorithmName': 'InitialdEdx', 'bins': np.linspace(0, 20000, num=100), 'cutDirection': 'left'},
)

featureHistogram = {
    "plot": True,
    "filters": (
        #("Showers", "isShower==1", "", True), 
        #("Tracks", "isShower==0", "", True),
        ("Electrons + Positrons", "abs(mcPdgCode)==11", "isShower==1", True),
        ("Photons", "abs(mcPdgCode)==22",  "isShower==1", True),
        #("Protons", "abs(mcPdgCode)==2212", "isShower==0", False),
        #("Muons", "abs(mcPdgCode)==13", "isShower==0", False),
        #("Charged Pions", "abs(mcPdgCode)==211", "isShower==0", False),
    )   
}
purityEfficiency = {
    "plot": True,
    "nTestCuts": 1000
}

# Load the pickle file.
ds.dfPerfPfoData = ds.GetFilteredPfoData(ds.performancePreFilters, features, portion=(ds.trainingFraction, 1))

print("Analysing features using the following samples:\n")
ds.PrintSampleInput(ds.dfPerfPfoData)

for feature in features:
    dfPfoData = ds.dfPerfPfoData["all"][ds.GetFeatureView(feature["name"])]
    dfTrackData = ds.dfPerfPfoData["track"][ds.GetFeatureView(feature["name"])]
    dfShowerData = ds.dfPerfPfoData["shower"][ds.GetFeatureView(feature["name"])]
    cutoffValues, cutoffResults = fa.GetBestPurityEfficiency(dfTrackData, dfShowerData, ("track", "shower"), feature, purityEfficiency["nTestCuts"])
    if featureHistogram["plot"]:
        fa.PlotVariableHistogram(dfPfoData, ("track", "shower"), feature, featureHistogram, cutoffResults[4])
    if purityEfficiency["plot"]:        
        fa.PlotPurityEfficiencyVsCutoff(feature["name"], ("track", "shower"), cutoffValues, cutoffResults)

fa.CorrelationMatrix([feature['name'] for feature in features], ds.GetFeatureViews(features), ds.performancePreFilters, ds.dfPerfPfoData)