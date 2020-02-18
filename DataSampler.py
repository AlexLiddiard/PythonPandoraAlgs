import BaseConfig as bc
import pandas as pd
import math as m
import numpy as np
import os
from UpRootFileReader import MicroBooneGeo
import GeneralConfig as gc
import DataSamplerConfig as cfg

dfInputPfoData = None
# Load pickle file
def LoadPfoData(features=None):
    global dfInputPfoData
    print("Loading data from pickle files")
    if features is not None:
        algorithmNames = GetFeatureAlgorithms(features)
        algorithmNames.pop("GeneralInfo", None)  
    else:
        algorithmNames = GetAllDataAlgorithms(bc.dataFolderFull, cfg.dataSources["all"])
    dfInputPfoData = []
    for dataName in cfg.dataSources["all"]:
        featureDataArray = [pd.read_pickle(bc.dataFolderFull + "/" + dataName + "_GeneralInfo.pickle")]
        for algorithmName, featureNames in algorithmNames.items():
            dfAlgorithm = pd.read_pickle(bc.dataFolderFull + "/" + dataName + "_" + algorithmName + ".pickle")
            featureDataArray.append(dfAlgorithm[featureNames])
        dfInputPfoData.append(pd.concat(featureDataArray, axis=1, sort=False))
        dfInputPfoData[-1]["dataName"] = dataName
    dfInputPfoData = pd.concat(dfInputPfoData, ignore_index=True)

def SavePfoData(dfData, algorithmName):
    position = 0
    for dataName in cfg.dataSources["all"]:
        length = len(dfInputPfoData.query("dataName==@dataName"))
        dfData[position:(position + length)].reset_index(drop=True).to_pickle(bc.dataFolderFull + "/" + dataName + "_" + algorithmName + ".pickle") # Undo dataframe concatenation, save the file
        position += length

def GetFilteredPfoData(dataSource, pfoClass, filterClass, filterName):
    # Data source filtering
    dfPfoData = []
    for dataName, portion in cfg.dataSources[dataSource].items():
        dfTemp = dfInputPfoData.query("dataName==@dataName").sample(frac=1, random_state=gc.random_state)
        if portion is not None:
            dfTemp = dfTemp[m.floor(len(dfTemp) * portion[0]):m.floor(len(dfTemp) * portion[1])]
        dfPfoData.append(dfTemp)
    dfPfoData = pd.concat(dfPfoData, ignore_index=True)    

    # Class filtering
    dfPfoData = dfPfoData.query(gc.classes[pfoClass])

    # View/other filtering
    if filterName != "unfiltered":
        if cfg.preFilters[filterClass]["general"] != "":
            dfPfoData = dfPfoData.query(cfg.preFilters[filterClass]["general"])
        if cfg.preFilters[filterClass][filterName] != "":
            dfPfoData = dfPfoData.query(cfg.preFilters[filterClass][filterName])
    dfPfoData = dfPfoData.reset_index(drop=True)

    print((
        "Got %s PFOs satisfying the following filter options" +
        "\n\tData source: %s" +
        "\n\tPFO class: %s" +
        "\n\tFilter class: %s" +
        "\n\tFilter name: %s"
    ) % (len(dfPfoData), dataSource, pfoClass, filterClass, filterName))
    return dfPfoData

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

# Find the names of all algorithm data sets, starting with a given base name.
# If multiple base names, it will return the intersection of the names from each.
def GetAllDataAlgorithms(dataFolder, dataNames):
    if len(dataNames) == 0:
        return []
    algorithmNames = {}
    for fileName in os.listdir(dataFolder):
        for dataName in dataNames:
            if os.path.isfile(os.path.join(dataFolder, fileName)) and fileName.startswith(dataName):
                algorithmName = os.path.splitext(fileName[len(dataName) + 1:])[0]
                algorithmNames[algorithmName] = algorithmNames.get(algorithmName, 0) + 1
    return dict.fromkeys([algorithmName for algorithmName in algorithmNames if algorithmNames[algorithmName] == len(dataNames)])


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