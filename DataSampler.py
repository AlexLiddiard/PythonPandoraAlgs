import pandas as pd
import math as m
import numpy as np
import BaseConfig as bc
from UpRootFileReader import MicroBooneGeo
import DataSamplerConfig as cfg
import GeneralConfig as gc

dfInputPfoData = None
# Load pickle file
def LoadPfoData(features):
    global dfInputPfoData
    print("Loading data from pickle files")
    algorithmNames = GetFeatureAlgorithms(features)
    algorithmNames['GeneralInfo'] = []
    dfInputPfoData = []
    for dataName in cfg.dataSources["all"]:
        featureDataArray = []
        for algorithmName in algorithmNames:
            dfAlgorithm = pd.read_pickle(bc.dataFolderFull+ "/" + dataName + "_" + algorithmName + ".pickle")
            if algorithmName == 'GeneralInfo':
                featureDataArray.append(dfAlgorithm)
            else:
                featureDataArray.append(dfAlgorithm[algorithmNames[algorithmName]])
        
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
    if pfoClass != "all":
        dfPfoData = dfPfoData.query(gc.classes[pfoClass])

    # View/other filtering
    if filterName == "unfiltered":
        return dfPfoData.reset_index(drop=True)
    dfPfoData = dfPfoData.query(cfg.preFilters[filterClass]["general"])
    return dfPfoData.query(cfg.preFilters[filterClass][filterName]).reset_index(drop=True)

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