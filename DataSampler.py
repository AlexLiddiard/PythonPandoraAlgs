import pandas as pd
import math as m
import numpy as np
from UpRootFileReader import MicroBooneGeo

myTestArea = "/home/tomalex/Pandora/"
dataFolder = myTestArea + '/PythonPandoraAlgs/TrackShowerData/'

# "DataSourceName": (data start fraction, data end fraction)
#  where e.g. data start fraction = (data start position) / (# of samples)
trainingDataSources = {
    "BNBNuOnly": (0, 0.5),
    "BNBNuOnly400-800": (0, 1),
    "BNBNuOnly0-400": (0, 1)
}
performanceDataSources = {
    "BNBNuOnly": (0.5, 1),
}
allDataSources = dict.fromkeys(list(trainingDataSources) + list(performanceDataSources))
random_state = 201746973
trainingFraction = 0.5
trainingPreFilters = {
    "general": (
        'abs(mcPdgCode) != 2112',
        #'minCoordX >= @MicroBooneGeo.RangeX[0] + 10',
        #'maxCoordX <= @MicroBooneGeo.RangeX[1] - 10',
        #'minCoordY >= @MicroBooneGeo.RangeY[0] + 20',
        #'maxCoordY <= @MicroBooneGeo.RangeY[1] - 20',
        #'minCoordZ >= @MicroBooneGeo.RangeY[0] + 10',
        #'maxCoordZ <= @MicroBooneGeo.RangeZ[1] - 10',
    ),
    "U": (
        '(isShower==1 and purityU>=0.8) or (isShower==0 and purityU>=0.8)',
        '(isShower==1 and completenessU>=0.5) or (isShower==0 and completenessU>=0.8)',
        #'nHitsU>=50',
    ),
    "V": (
        '(isShower==1 and purityV>=0.8) or (isShower==0 and purityV>=0.8)',
        '(isShower==1 and completenessV>=0.5) or (isShower==0 and completenessV>=0.8)',
        #'nHitsV>=50',
    ),
    "W": (
        '(isShower==1 and purityW>=0.8) or (isShower==0 and purityW>=0.8)',
        '(isShower==1 and completenessW>=0.5) or (isShower==0 and completenessW>=0.8)',
        #'nHitsW>=50',
    ),
    "3D":
    (
        '(isShower==1 and purityU>=0.8) or (isShower==0 and purityU>=0.8)',
        '(isShower==1 and purityV>=0.8) or (isShower==0 and purityV>=0.8)',
        '(isShower==1 and purityW>=0.8) or (isShower==0 and purityW>=0.8)',
        '(isShower==1 and completenessU>=0.5) or (isShower==0 and completenessU>=0.8)',
        '(isShower==1 and completenessV>=0.5) or (isShower==0 and completenessV>=0.8)',
        '(isShower==1 and completenessW>=0.5) or (isShower==0 and completenessW>=0.8)',
        #'nHits3D >= 50',
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

dfInputPfoData = None
# Load pickle file
def LoadPfoData(features):
    global dfInputPfoData
    print("Loading data from pickle files")
    algorithmNames = GetFeatureAlgorithms(features)
    algorithmNames['GeneralInfo'] = []
    dfInputPfoData = {}
    for dataName in allDataSources:
        featureDataArray = []
        for algorithmName in algorithmNames:
            dfAlgorithm = pd.read_pickle(dataFolder + dataName + "_" + algorithmName + ".pickle")
            if algorithmName == 'GeneralInfo':
                featureDataArray.append(dfAlgorithm)
            else:
                featureDataArray.append(dfAlgorithm[algorithmNames[algorithmName]])
        
        dfInputPfoData[dataName] = pd.concat(featureDataArray, axis=1, sort=False)
    dfInputPfoData["all"] = pd.concat(dfInputPfoData.values(), ignore_index=True)
    #np.random.seed(random_state)

def SavePfoData(df, algorithmName):
    position = 0
    for dataName in allDataSources:
        length = len(dfInputPfoData[dataName])
        df[position:(position + length)].reset_index(drop=True).to_pickle(dataFolder + dataName + "_" + algorithmName + ".pickle") # Undo dataframe concatenation, save the file
        position += length

def GetFilteredDataframes(df, filters):
    dfs = {}
    for key in filters:
        dfs[key] = df.query(filters[key]).copy().reset_index(drop=True)
    dfs["union"] = df.query("(" + ") or (".join(filters.values()) + ")").copy().reset_index(drop=True)
    dfs["intersection"] = df.query("(" + ") and (".join(filters.values()) + ")").copy().reset_index(drop=True)
    return dfs

def GetFilteredPfoData(dataSources, filters, features):
    if dfInputPfoData is None:
        LoadPfoData(features)
    usedViewFilters = {key:filters[key] for key in GetFeatureViews(features)}

    tempData = []
    for dataName in dataSources:
        dfTemp = dfInputPfoData[dataName]
        np.random.seed(random_state)
        dfTemp = dfTemp.sample(frac=1, random_state=random_state)
        portion = dataSources[dataName]
        if portion is not None:
            dfTemp = dfTemp[m.floor(len(dfTemp) * portion[0]):m.floor(len(dfTemp) * portion[1])].copy().reset_index(drop=True)
        tempData.append(dfTemp)
    
    filteredPfoData = {"all": {}, "track": {}, "shower": {}}
    filteredPfoData["all"]["unfiltered"] = pd.concat(tempData)
    filteredPfoData["all"]["general"] = filteredPfoData["all"]["unfiltered"].query(filters["general"]).copy().reset_index(drop=True)
    filteredPfoData["all"].update(GetFilteredDataframes(filteredPfoData["all"]["general"], usedViewFilters))
    filteredPfoData["shower"]["general"] = filteredPfoData["all"]["general"].query("isShower==1").copy().reset_index(drop=True)
    filteredPfoData["shower"].update(GetFilteredDataframes(filteredPfoData["shower"]["general"], usedViewFilters))
    filteredPfoData["track"]["general"] = filteredPfoData["all"]["general"].query("isShower==0").copy().reset_index(drop=True)
    filteredPfoData["track"].update(GetFilteredDataframes(filteredPfoData["track"]["general"], usedViewFilters))
    return filteredPfoData

dfTrainingPfoData = None
def GetTrainingPfoData(features):
    global dfTrainingPfoData
    print("Getting training PFOs")
    dfTrainingPfoData = GetFilteredPfoData(filters=trainingPreFilters, features=features, dataSources=trainingDataSources)

dfPerfPfoData = None
def GetPerfPfoData(features):
    global dfPerfPfoData
    print("Getting performance PFOs")
    dfPerfPfoData = GetFilteredPfoData(filters=performancePreFilters, features=features, dataSources=performanceDataSources)

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

def GetFeatureNames(features):
    return [feature['name'] for feature in features]

def GetFeatureViews(features):
    viewsUsed = {}
    for feature in features:
        view = GetFeatureView(feature["name"])
        if view not in viewsUsed:
            viewsUsed[view] = []
        viewsUsed[view].append(feature["name"])
    return viewsUsed

def PrintSampleInput(pfoData):
    for className in pfoData:
        print(className + ":")
        for view in pfoData[className]:
            print("\t%s: %s PFOs" % (view, len(pfoData[className][view])))

# Initialise prefilters
ProcessFilters(trainingPreFilters)
ProcessFilters(performancePreFilters)