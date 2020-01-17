from sklearn import tree
from sklearn import ensemble
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
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
    {'name': 'RSquaredU'},
    {'name': 'RSquaredV'},
    {'name': 'RSquaredW'},
    {'name': 'BinnedHitStdU'},
    {'name': 'BinnedHitStdV'},
    {'name': 'BinnedHitStdW'},
    {'name': 'RadialBinStdU'},
    {'name': 'RadialBinStdV'},
    {'name': 'RadialBinStdW'},
    {'name': 'RadialBinStd3D'},
    {'name': 'ChainCountU'},
    {'name': 'ChainCountV'},
    {'name': 'ChainCountW'},
    {'name': 'ChainRatioAvgU'},
    {'name': 'ChainRatioAvgV'},
    {'name': 'ChainRatioAvgW'},
    {'name': 'ChainRSquaredAvgU'},
    {'name': 'ChainRSquaredAvgV'},
    {'name': 'ChainRSquaredAvgW'},
    {'name': 'ChainRatioStdU'},
    {'name': 'ChainRatioStdV'},
    {'name': 'ChainRatioStdW'},
    {'name': 'ChainRSquaredStdU'},
    {'name': 'ChainRSquaredStdV'},
    {'name': 'ChainRSquaredStdW'},
    {'name': 'AngularSpanU'},
    {'name': 'AngularSpanV'},
    {'name': 'AngularSpanW'},
    {'name': 'AngularSpan3D'},
    {'name': 'LongitudinalSpanU'},
    {'name': 'LongitudinalSpanV'},
    {'name': 'LongitudinalSpanW'},
    {'name': 'LongitudinalSpan3D'},
    {'name': 'PcaVarU'},
    {'name': 'PcaVarV'},
    {'name': 'PcaVarW'},
    {'name': 'PcaVar3D'},
    {'name': 'PcaRatioU'},
    {'name': 'PcaRatioV'},
    {'name': 'PcaRatioW'},
    {'name': 'PcaRatio3D'},
    {'name': 'ChargedBinnedHitStdU'},
    {'name': 'ChargedBinnedHitStdV'},
    {'name': 'ChargedBinnedHitStdW'},
    {'name': 'ChargedStdMeanRatioU'},
    {'name': 'ChargedStdMeanRatioV'},
    {'name': 'ChargedStdMeanRatioW'},
    {'name': 'BraggPeakU'},
    {'name': 'BraggPeakV'},
    {'name': 'BraggPeakW'},
    {'name': 'BraggPeak3D'},
    {'name': 'Moliere3D'}
)

# Load the training PFOs
ds.GetTrainingPfoData()

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
    imp = IterativeImputer()
    imp.fit(trainingDataFeed)
    trainingDataFeed = imp.transform(trainingDataFeed)
    #ro = ros(ratio=1)
    sm = smt(ratio='minority')
    #smt = smtmk(ratio=1, )
    trainingDataFeed, classificationArray = sm.fit_sample(trainingDataFeed, classificationArray)
    clfView = clf.fit(trainingDataFeed, classificationArray)
    return clfView.decision_function(ds.dfAllPfoData[featureNames])

print("Calculating BDT values for each view")
clf = ensemble.HistGradientBoostingClassifier()
viewsUsed = ds.GetViewsUsed(features)
for view in viewsUsed:
    ds.dfAllPfoData["BDT" + view] = GetBDTValues(clf, view)

ds.GetTrainingPfoData()
print("Calculating multi-view BDT values")
valueMask = {
    "BDTU": ds.trainingPreFilters["U"],
    "BDTV": ds.trainingPreFilters["V"],
    "BDTW": ds.trainingPreFilters["W"],
    "BDT3D": ds.trainingPreFilters["3D"]
}
ds.dfAllPfoData["BDTMulti"] = GetBDTValues(clf, 'union')#, valueMask)

ds.SavePickleFile()
print("Finished!")