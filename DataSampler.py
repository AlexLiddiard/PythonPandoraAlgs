import pandas as pd
import math as m
from UpRootFileReader import MicroBooneGeo

myTestArea = "/home/tomalex/Pandora/"
dataFolder = myTestArea + '/PythonPandoraAlgs/TrackShowerData/'
dataName = "BNBNuOnly"

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
        '(isShower==1 and purityU>=0.8) or (isShower==0 and purityU>=0.8)',
        '(isShower==1 and completenessU>=0.5) or (isShower==0 and completenessU>=0.8)',
        'nHitsU>=50',
    ),
    "V": (
        '(isShower==1 and purityV>=0.8) or (isShower==0 and purityV>=0.8)',
        '(isShower==1 and completenessV>=0.5) or (isShower==0 and completenessV>=0.8)',
        'nHitsV>=50',
    ),
    "W": (
        '(isShower==1 and purityW>=0.8) or (isShower==0 and purityW>=0.8)',
        '(isShower==1 and completenessW>=0.5) or (isShower==0 and completenessW>=0.8)',
        'nHitsW>=50',
    ),
    "3D":
    (
        '(isShower==1 and purityU>=0.8) or (isShower==0 and purityU>=0.8)',
        '(isShower==1 and purityV>=0.8) or (isShower==0 and purityV>=0.8)',
        '(isShower==1 and purityW>=0.8) or (isShower==0 and purityW>=0.8)',
        '(isShower==1 and completenessU>=0.5) or (isShower==0 and completenessU>=0.8)',
        '(isShower==1 and completenessV>=0.5) or (isShower==0 and completenessV>=0.8)',
        '(isShower==1 and completenessW>=0.5) or (isShower==0 and completenessW>=0.8)',
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

def ProcessFilters(filters):
    for key in filters:
        filters[key] = ' and '.join(filters[key])
    viewFilters = [filters[x] for x in ["U", "V", "W", "3D"]]
    filters["union"] = "(" + ") or (".join(viewFilters) + ")"
    filters["intersection"] = "(" + ") and (".join(viewFilters) + ")"

dfAllPfoData = None
# Load pickle file
def LoadPfoData(features):
    global dfAllPfoData
    print("Loading pickle file")
    algorithmNames = GetFeatureAlgorithms(features)
    algorithmNames['GeneralInfo'] = []
    featureDataArray = []
    for algorithmName in algorithmNames:
        dfAlgorithm = pd.read_pickle(dataFolder + dataName + "_" + algorithmName + ".pickle")
        if algorithmName == 'GeneralInfo':
            featureDataArray.append(dfAlgorithm)
        else:
            featureDataArray.append(dfAlgorithm[algorithmNames[algorithmName]])
    dfAllPfoData = pd.concat(featureDataArray, axis=1, sort=False)

def SavePfoData(df, algorithmName):
    df.to_pickle(dataFolder + dataName + "_" + algorithmName + ".pickle")

def GetFilteredDataframes(df, filters):
    dfs = {}
    for key in filters:
        dfs[key] = df.query(filters[key]).copy()
    dfs["union"] = df.query("(" + ") or (".join(filters.values()) + ")").copy()
    dfs["intersection"] = df.query("(" + ") and (".join(filters.values()) + ")").copy()
    return dfs

def GetFilteredPfoData(filters, features, portion=(0, 1)):
    if dfAllPfoData is None:
        LoadPfoData(features)
    usedViewFilters = {key:filters[key] for key in GetViewsUsed(features)}
    filteredPfoData = {}
    filteredPfoData["all"] = {"unfiltered":dfAllPfoData[m.floor(len(dfAllPfoData) * portion[0]):m.floor(len(dfAllPfoData) * portion[1])].copy()}
    filteredPfoData["all"]["general"] = filteredPfoData["all"]["unfiltered"].query(filters["general"]).copy()
    filteredPfoData["all"].update(GetFilteredDataframes(filteredPfoData["all"]["general"], usedViewFilters))
    filteredPfoData["shower"] = {"general": filteredPfoData["all"]["general"].query("isShower==1").copy()}
    filteredPfoData["shower"].update(GetFilteredDataframes(filteredPfoData["shower"]["general"], usedViewFilters))
    filteredPfoData["track"] = {"general": filteredPfoData["all"]["general"].query("isShower==0").copy()}
    filteredPfoData["track"].update(GetFilteredDataframes(filteredPfoData["track"]["general"], usedViewFilters))
    return filteredPfoData

dfTrainingPfoData = None
def GetTrainingPfoData(features):
    global dfTrainingPfoData
    print("Getting training PFOs")
    dfTrainingPfoData = GetFilteredPfoData(filters=trainingPreFilters, features=features, portion=(0, trainingFraction))

dfPerfPfoData = None
def GetPerfPfoData(features):
    global dfPerfPfoData
    print("Getting performance PFOs")
    dfPerfPfoData = GetFilteredPfoData(filters=performancePreFilters, features=features, portion=(trainingFraction, 1))

def GetFeatureView(featureName):
    if featureName.endswith("3D"):
        return "3D"
    if featureName[-1] in ["U", "V", "W"]:
        return featureName[-1]
    return "union"

def GetFeatureAlgorithms(features):
    algorithmNames = {}
    for feature in features:
        if feature['algorithmName'] not in algorithmNames:
            algorithmNames[feature['algorithmName']] = []
        algorithmNames[feature['algorithmName']].append(feature['name'])
    return algorithmNames

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

def PrintSampleInput(pfoData):
    for className in pfoData:
        print(className + ":")
        for view in pfoData[className]:
            print("\t%s: %s PFOs" % (view, len(pfoData[className][view])))

# Initialise prefilters
ProcessFilters(trainingPreFilters)
ProcessFilters(performancePreFilters)