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
    {'name': 'ChargedBinnedHitStdU', 'algorithmName': 'ChargeHitBinning'},
    {'name': 'ChargedBinnedHitStdV', 'algorithmName': 'ChargeHitBinning'},
    {'name': 'ChargedBinnedHitStdW', 'algorithmName': 'ChargeHitBinning'},
    {'name': 'ChargedStdMeanRatioU', 'algorithmName': 'ChargeHitBinning'},
    {'name': 'ChargedStdMeanRatioV', 'algorithmName': 'ChargeHitBinning'},
    {'name': 'ChargedStdMeanRatioW', 'algorithmName': 'ChargeHitBinning'},
    {'name': 'ChargedStdMeanRatio3D', 'algorithmName': 'ChargeHitBinning'},
    {'name': 'BraggPeakU', 'algorithmName': 'BraggPeak'},
    {'name': 'BraggPeakV', 'algorithmName': 'BraggPeak'},
    {'name': 'BraggPeakW', 'algorithmName': 'BraggPeak'},
    {'name': 'BraggPeak3D', 'algorithmName': 'BraggPeak'},
    {'name': 'Moliere3D', 'algorithmName': 'MoliereRadius'},
)

# Load the training PFOs
ds.GetTrainingPfoData(features)

print("Training BDTs using the following samples:")
for view in ds.dfTrainingPfoData["track"]:
    print("%s: %s tracks, %s showers" % (view, len(ds.dfTrainingPfoData["track"][view]), len(ds.dfTrainingPfoData["shower"][view])))
    

def GetViewFeatures(features, view):
    viewFeatures = []
    for feature in features:
        if ds.GetFeatureView(feature['name']) == view:
            viewFeatures.append(feature)
    return viewFeatures


def GetBDTValues(clf, view):
    if view == "union":
        featureNames = ["BDT3D", "BDTU", "BDTV", "BDTW"]
    else:
        viewFeatures = GetViewFeatures(features, view)
        featureNames = [feature['name'] for feature in viewFeatures]
    trainingDataFeed = ds.dfTrainingPfoData['all'][view]
    classificationArray = trainingDataFeed.eval("isShower==0")
    trainingDataFeed = trainingDataFeed[featureNames] # Remove all irrelevant columns
    imp = IterativeImputer(estimator=ensemble.HistGradientBoostingRegressor())
    imp.fit(trainingDataFeed)
    trainingDataFeed = imp.transform(trainingDataFeed)
    sm = smt(sampling_strategy=1)
    trainingDataFeed, classificationArray = sm.fit_sample(trainingDataFeed, classificationArray)
    clfView = clf.fit(trainingDataFeed, classificationArray)
    return clfView.decision_function(ds.dfAllPfoData[featureNames])

print("Calculating BDT values for each view")
clf = ensemble.HistGradientBoostingClassifier()
viewsUsed = ds.GetViewsUsed(features)
bdtData = {}
for view in viewsUsed:
    bdtData["BDT" + view] = GetBDTValues(clf, view)

ds.GetTrainingPfoData(features)
print("Calculating multi-view BDT values")
valueMask = {
    "BDTU": ds.trainingPreFilters["U"],
    "BDTV": ds.trainingPreFilters["V"],
    "BDTW": ds.trainingPreFilters["W"],
    "BDT3D": ds.trainingPreFilters["3D"]
}
bdtData["BDTMulti"] = GetBDTValues(clf, 'union')
ds.SavePfoData(pd.DataFrame(bdtData), "DecisionTreeCalculator")
print("Finished!")