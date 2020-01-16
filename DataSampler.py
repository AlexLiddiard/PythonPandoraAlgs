import pandas as pd
import math as m
from UpRootFileReader import MicroBooneGeo

myTestArea = "/home/jack/Documents/Pandora/"
inputPickleFile = myTestArea + '/PythonPandoraAlgs/featureData.bz2'
outputPickleFile = myTestArea + '/PythonPandoraAlgs/featureData.bz2'

trainingFraction = 0.5
trainingPreFilters = {
    "general": (
        'abs(mcPdgCode) != 2112',
        'minCoordX >= @MicroBooneGeo.RangeX[0] + 10',
        'maxCoordX <= @MicroBooneGeo.RangeX[1] - 10',
        'minCoordY >= @MicroBooneGeo.RangeY[0] + 20',
        'maxCoordY <= @MicroBooneGeo.RangeY[1] - 20',
        'minCoordZ >= @MicroBooneGeo.RangeY[0] + 10',
        'maxCoordZ <= @MicroBooneGeo.RangeZ[1] - 10',
    ),
    "U": (
        'purityU>=0.8',
        'completenessU>=0.8',
        'nHitsU>=50',
    ),
    "V": (
        'purityV>=0.8',
        'completenessV>=0.8',
        'nHitsV>=50',
    ),
    "W": (
        'purityW>=0.8',
        'completenessW>=0.8',
        'nHitsW>=50',
    ),
    "3D":
    (
        'purityU>=0.8 and purityV>=0.8 and purityW>=0.8',
        'completenessU>=0.8 and completenessV>=0.8 and completenessW>=0.8',
        'nHits3D >= 50',
    )
}

performancePreFilters = {
    "general": (
        'abs(mcPdgCode) != 2112',
        'minCoordX >= @MicroBooneGeo.RangeX[0] + 10',
        'maxCoordX <= @MicroBooneGeo.RangeX[1] - 10',
        'minCoordY >= @MicroBooneGeo.RangeY[0] + 20',
        'maxCoordY <= @MicroBooneGeo.RangeY[1] - 20',
        'minCoordZ >= @MicroBooneGeo.RangeY[0] + 10',
        'maxCoordZ <= @MicroBooneGeo.RangeZ[1] - 10',
        'nHitsU>=20 and nHitsV >= 20 and nHitsW>=20 and nHits3D>=20'
        #"nHitsU + nHitsV + nHitsW >= 100"
    ),
    "U": (
        #'purityU>=0.8',
        #'completenessU>=0.8',
        'nHitsU>=20',
    ),
    "V": (
        #'purityV>=0.8',
        #'completenessV>=0.8',
        'nHitsV>=20',
    ),
    "W": (
        #'purityW>=0.8',
        #'completenessW>=0.8',
        'nHitsW>=20',
    ),
    "3D":
    (
        #'purityU>=0.8 and purityV>=0.8 and purityW>=0.8',
        #'completenessU>=0.8 and completenessV>=0.8 and completenessW>=0.8',
        'nHits3D>=20',
    )
}

# Get training pre-filters, determine views used.
trainingPreFilters["general"] = ' and '.join(trainingPreFilters["general"])
trainingPreFilters["U"] = ' and '.join(trainingPreFilters["U"])
trainingPreFilters["V"] = ' and '.join(trainingPreFilters["V"])
trainingPreFilters["W"] = ' and '.join(trainingPreFilters["W"])
trainingPreFilters["3D"] = ' and '.join(trainingPreFilters["3D"])
viewFilters = [trainingPreFilters[x] for x in ["U", "V", "W", "3D"]]
trainingPreFilters["union"] = "(" + ") or (".join(viewFilters) + ")"
trainingPreFilters["intersection"] = "(" + ") and (".join(viewFilters) + ")"

# Get performance pre-filters.
performancePreFilters["general"] = ' and '.join(performancePreFilters["general"])
performancePreFilters["U"] = ' and '.join(performancePreFilters["U"])
performancePreFilters["V"] = ' and '.join(performancePreFilters["V"])
performancePreFilters["W"] = ' and '.join(performancePreFilters["W"])
performancePreFilters["3D"] = ' and '.join(performancePreFilters["3D"])
viewFilters = [performancePreFilters[x] for x in ["U", "V", "W", "3D"]]
performancePreFilters["union"] = "(" + ") or (".join(viewFilters) + ")"
performancePreFilters["intersection"] = "(" + ") and (".join(viewFilters) + ")"

dfPfoData = None
nPfoData = 0
# Load pickle file
def LoadPickleFile():
    global dfPfoData
    global nPfoData
    print("Loading pickle file")
    dfPfoData = pd.read_pickle(inputPickleFile)
    nPfoData = len(dfPfoData)

def SavePickleFile():
    print("Saving pickle file")
    dfPfoData.to_pickle(outputPickleFile)

# Get PFOs
dfTrainingPfoData = None
def GetTrainingPfoData(preFilters=trainingPreFilters, trainingFraction=trainingFraction):
    global dfTrainingPfoData
    if dfPfoData is None:
        LoadPickleFile()
    print("Getting training PFOs")
    dfTrainingPfoData = dfPfoData[:m.floor(nPfoData * trainingFraction)]
    dfTrainingPfoData = {"general": dfTrainingPfoData.query(preFilters["general"])}
    dfTrainingPfoData["shower"] = {"general": dfTrainingPfoData["general"].query("isShower==1")}
    dfTrainingPfoData["shower"]["U"] = dfTrainingPfoData["shower"]["general"].query(preFilters["U"])
    dfTrainingPfoData["shower"]["V"] = dfTrainingPfoData["shower"]["general"].query(preFilters["V"])
    dfTrainingPfoData["shower"]["W"] = dfTrainingPfoData["shower"]["general"].query(preFilters["W"])
    dfTrainingPfoData["shower"]["3D"] = dfTrainingPfoData["shower"]["general"].query(preFilters["3D"])
    dfTrainingPfoData["shower"]["union"] = dfTrainingPfoData["shower"]["general"].query(preFilters["union"])
    dfTrainingPfoData["shower"]["intersection"] = dfTrainingPfoData["shower"]["general"].query(preFilters["intersection"])
    dfTrainingPfoData["track"] = {"general": dfTrainingPfoData["general"].query("isShower==0")}
    dfTrainingPfoData["track"]["U"] = dfTrainingPfoData["track"]["general"].query(preFilters["U"])
    dfTrainingPfoData["track"]["V"] = dfTrainingPfoData["track"]["general"].query(preFilters["V"])
    dfTrainingPfoData["track"]["W"] = dfTrainingPfoData["track"]["general"].query(preFilters["W"])
    dfTrainingPfoData["track"]["3D"] = dfTrainingPfoData["track"]["general"].query(preFilters["3D"])
    dfTrainingPfoData["track"]["union"] = dfTrainingPfoData["track"]["general"].query(preFilters["union"])
    dfTrainingPfoData["track"]["intersection"] = dfTrainingPfoData["track"]["general"].query(preFilters["intersection"])

# Get performance PFOs
dfPerfPfoData = None
nPerfPfoData = None
def GetPerformancePfoData(preFilters=performancePreFilters, viewsUsed=["U", "V", "W", "3D"], trainingFraction = trainingFraction):
    global dfPerfPfoData
    global nPerfPfoData
    if dfPfoData is None:
        LoadPickleFile()
    print("Getting performance PFOs")
    dfPerfPfoData = dfPfoData[m.floor(nPfoData * trainingFraction):].query(preFilters["general"])
    viewFilters = []
    for key in viewsUsed:
        viewFilters.append(preFilters[key])
    performanceFilter = "(" + ") or (".join(viewFilters) + ")"
    dfPerfPfoData = {"general": dfPerfPfoData.query(performanceFilter).reset_index(drop=True)}
    dfPerfPfoData["shower"] = dfPerfPfoData["general"].query("isShower==1")
    dfPerfPfoData["track"] = dfPerfPfoData["general"].query("isShower==0")
    nPerfPfoData = {
        "general": len(dfPerfPfoData["general"]),
        "shower": len(dfPerfPfoData["shower"]),
        "track": len(dfPerfPfoData["track"])
    }


def GetFeatureView(featureName):
    if featureName.endswith("3D"):
        return "3D"
    if featureName[-1] in ["U", "V", "W"]:
        return featureName[-1]
    return "union"

def GetViewFeatures(features, view):
    viewFeatures = []
    for feature in features:
        if GetFeatureView(feature['name']) == view:
            viewFeatures.append(feature)
    return viewFeatures

def GetViewsUsed(features):
    viewsUsed = []
    for feature in features:
        viewsUsed.append(GetFeatureView(feature["name"]))
    return list(dict.fromkeys(viewsUsed))