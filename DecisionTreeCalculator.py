from sklearn import tree
from sklearn import ensemble
from sklearn.experimental import enable_hist_gradient_boosting
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math as m
from UpRootFileReader import MicroBooneGeo
import DataSampler as ds
from imblearn.combine import SMOTETomek as smtmk
from imblearn.over_sampling import SMOTE as smt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from imblearn.over_sampling import RandomOverSampler as ros

features = (
    {'name': 'RSquaredU', 'algorithmName': 'LinearRegression'},
    {'name': 'RSquaredV', 'algorithmName': 'LinearRegression'},
    {'name': 'RSquaredW', 'algorithmName': 'LinearRegression'},
    {'name': 'BinnedHitStdU', 'algorithmName': 'HitBinning'},
    {'name': 'BinnedHitStdV', 'algorithmName': 'HitBinning'},
    {'name': 'BinnedHitStdW', 'algorithmName': 'HitBinning'},
    {'name': 'RadialBinStdU', 'algorithmName': 'HitBinning'},
    {'name': 'RadialBinStdV', 'algorithmName': 'HitBinning'},
    {'name': 'RadialBinStdW', 'algorithmName': 'HitBinning'},
    {'name': 'RadialBinStd3D', 'algorithmName': 'HitBinning'},
    {'name': 'ChainCountU', 'algorithmName': 'ChainCreation'},
    {'name': 'ChainCountV', 'algorithmName': 'ChainCreation'},
    {'name': 'ChainCountW', 'algorithmName': 'ChainCreation'},
    {'name': 'ChainRatioAvgU', 'algorithmName': 'ChainCreation'},
    {'name': 'ChainRatioAvgV', 'algorithmName': 'ChainCreation'},
    {'name': 'ChainRatioAvgW', 'algorithmName': 'ChainCreation'},
    {'name': 'ChainRSquaredAvgU', 'algorithmName': 'ChainCreation'},
    {'name': 'ChainRSquaredAvgV', 'algorithmName': 'ChainCreation'},
    {'name': 'ChainRSquaredAvgW', 'algorithmName': 'ChainCreation'},
    {'name': 'ChainRatioStdU', 'algorithmName': 'ChainCreation'},
    {'name': 'ChainRatioStdV', 'algorithmName': 'ChainCreation'},
    {'name': 'ChainRatioStdW', 'algorithmName': 'ChainCreation'},
    {'name': 'ChainRSquaredStdU', 'algorithmName': 'ChainCreation'},
    {'name': 'ChainRSquaredStdV', 'algorithmName': 'ChainCreation'},
    {'name': 'ChainRSquaredStdW', 'algorithmName': 'ChainCreation'},
    {'name': 'AngularSpanU', 'algorithmName': 'AngularSpan'},
    {'name': 'AngularSpanV', 'algorithmName': 'AngularSpan'},
    {'name': 'AngularSpanW', 'algorithmName': 'AngularSpan'},
    {'name': 'AngularSpan3D', 'algorithmName': 'AngularSpan'},
    {'name': 'LongitudinalSpanU', 'algorithmName': 'AngularSpan'},
    {'name': 'LongitudinalSpanV', 'algorithmName': 'AngularSpan'},
    {'name': 'LongitudinalSpanW', 'algorithmName': 'AngularSpan'},
    {'name': 'LongitudinalSpan3D', 'algorithmName': 'AngularSpan'},
    {'name': 'PcaVarU', 'algorithmName': 'PCAnalysis'},
    {'name': 'PcaVarV', 'algorithmName': 'PCAnalysis'},
    {'name': 'PcaVarW', 'algorithmName': 'PCAnalysis'},
    {'name': 'PcaVar3D', 'algorithmName': 'PCAnalysis'},
    {'name': 'PcaRatioU', 'algorithmName': 'PCAnalysis'},
    {'name': 'PcaRatioV', 'algorithmName': 'PCAnalysis'},
    {'name': 'PcaRatioW', 'algorithmName': 'PCAnalysis'},
    {'name': 'PcaRatio3D', 'algorithmName': 'PCAnalysis'},
    {'name': 'ChargedBinnedHitStdU', 'algorithmName': 'ChargedHitBinning'},
    {'name': 'ChargedBinnedHitStdV', 'algorithmName': 'ChargedHitBinning'},
    {'name': 'ChargedBinnedHitStdW', 'algorithmName': 'ChargedHitBinning'},
    {'name': 'ChargedStdMeanRatioU', 'algorithmName': 'ChargeStdMeanRatio'},
    {'name': 'ChargedStdMeanRatioV', 'algorithmName': 'ChargeStdMeanRatio'},
    {'name': 'ChargedStdMeanRatioW', 'algorithmName': 'ChargeStdMeanRatio'},
    {'name': 'ChargedStdMeanRatio3D', 'algorithmName': 'ChargeStdMeanRatio'},
    {'name': 'BraggPeakU', 'algorithmName': 'BraggPeak'},
    {'name': 'BraggPeakV', 'algorithmName': 'BraggPeak'},
    {'name': 'BraggPeakW', 'algorithmName': 'BraggPeak'},
    {'name': 'BraggPeak3D', 'algorithmName': 'BraggPeak'},
    {'name': 'Moliere3D', 'algorithmName': 'MoliereRadius'},
)

def TrainBDT(clf, featureNames, trainingData):
    classificationArray = trainingData.eval("isShower==0")
    trainingData = trainingData[featureNames] # Remove all irrelevant columns
    imp = IterativeImputer()
    imp.fit(trainingData)
    trainingData = imp.transform(trainingData)
    sm = smt(sampling_strategy=1)
    trainingData, classificationArray = sm.fit_sample(trainingData, classificationArray)
    clfView = clf.fit(trainingData, classificationArray)
    return clfView

def GetBDTValues(bdts, featureViews, evalData):
    btdValues = {}
    for btdName in bdts:
        view = ds.GetFeatureView(btdName)
        btdValues[btdName] = bdts[btdName].decision_function(evalData[featureViews[view]])
    return pd.DataFrame(btdValues)

# Load the training PFOs
ds.GetTrainingPfoData(features)

print("\nTraining BDTs using the following samples:")
ds.PrintSampleInput(ds.dfTrainingPfoData)

print("\nCalculating BDT values for each view")
featureViews = ds.GetFeatureViews(features)
bdts = {}

for view in featureViews:
    clf = ensemble.HistGradientBoostingClassifier()
    bdts["BDT" + view] = TrainBDT(clf, featureViews[view], ds.dfTrainingPfoData["all"][view])

dfBdtValues = GetBDTValues(bdts, featureViews, ds.dfTrainingPfoData["all"]["union"])
ds.dfTrainingPfoData["all"]["union"] = pd.concat([ds.dfTrainingPfoData["all"]["union"], dfBdtValues], axis=1, sort=False)

print("\nCalculating multi-view BDT values")
clf = ensemble.HistGradientBoostingClassifier()
bdtMulti = TrainBDT(clf, dfBdtValues.columns, ds.dfTrainingPfoData["all"]["union"])
dfBdtValues = GetBDTValues(bdts, featureViews, ds.dfAllPfoData)
dfBdtValues["BDTMulti"] = bdtMulti.decision_function(dfBdtValues)

print("\nSaving results")

ds.SavePfoData(dfBdtValues, "DecisionTreeCalculator")
print("Finished!")