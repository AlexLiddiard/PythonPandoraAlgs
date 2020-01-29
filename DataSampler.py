import pandas as pd
import math as m
import numpy as np
from UpRootFileReader import MicroBooneGeo

random_state = 201746973
myTestArea = "/home/tomalex/Pandora/"

def ProcessFilters(filters):
    for key in filters:
        filters[key] = ' and '.join(filters[key])
    viewFilters = [filters[x] for x in ["U", "V", "W", "3D"]]
    filters["union"] = "(" + ") or (".join(viewFilters) + ")"
    filters["intersection"] = "(" + ") and (".join(viewFilters) + ")"
    return filters

dfInputPfoData = None
# Load pickle file
def LoadPfoData(dataFolder, allDataSources, features):
    global dfInputPfoData
    print("Loading data from pickle files")
    algorithmNames = GetFeatureAlgorithms(features)
    algorithmNames['GeneralInfo'] = []
    dfInputPfoData = []
    for dataName in allDataSources:
        featureDataArray = []
        for algorithmName in algorithmNames:
            dfAlgorithm = pd.read_pickle(dataFolder + dataName + "_" + algorithmName + ".pickle")
            if algorithmName == 'GeneralInfo':
                featureDataArray.append(dfAlgorithm)
            else:
                featureDataArray.append(dfAlgorithm[algorithmNames[algorithmName]])
        
        dfInputPfoData.append(pd.concat(featureDataArray, axis=1, sort=False))
        dfInputPfoData[-1]["dataName"] = dataName
    dfInputPfoData = pd.concat(dfInputPfoData, ignore_index=True)

def SavePfoData(dataFolder, allDataSources, dfData, algorithmName):
    position = 0
    for dataName in allDataSources:
        length = len(dfInputPfoData[dataName])
        dfData[position:(position + length)].reset_index(drop=True).to_pickle(dataFolder + dataName + "_" + algorithmName + ".pickle") # Undo dataframe concatenation, save the file
        position += length

def GetFilteredPfoData(dataSources, classQueries, filters, className, filterName):
    # Data source filtering
    dfPfoData = []
    for dataName in dataSources:
        dfTemp = dfInputPfoData.query("dataName==@dataName").sample(frac=1, random_state=random_state)
        portion = dataSources[dataName]
        if portion is not None:
            dfTemp = dfTemp[m.floor(len(dfTemp) * portion[0]):m.floor(len(dfTemp) * portion[1])]
        dfPfoData.append(dfTemp)
    dfPfoData = pd.concat(dfPfoData, ignore_index=True)    

    # Class filtering
    if className != "all":
        dfPfoData = dfPfoData.query(classQueries[className])

    # View/other filtering
    if filterName == "unfiltered":
        return dfPfoData.reset_index(drop=True)
    dfPfoData = dfPfoData.query(filters["general"])
    if filterName == "union":
        return dfPfoData.query("(" + ") or (".join(filters.values()) + ")").reset_index(drop=True)
    if filterName == "intersection":
        return dfPfoData.query("(" + ") and (".join(filters.values()) + ")").reset_index(drop=True)
    return dfPfoData.query(filters[filterName]).reset_index(drop=True)

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