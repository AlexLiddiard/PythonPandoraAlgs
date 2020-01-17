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

print((
    "Training BDTs using the following samples:\n" +
    "General: %s tracks, %s showers\n" +
    "U View: %s tracks, %s showers\n" +
    "V View: %s tracks, %s showers\n" +
    "W View: %s tracks, %s showers\n" +
    "3D View: %s tracks, %s showers\n") %
    (
        len(ds.dfTrainingPfoData["track"]["general"]), len(ds.dfTrainingPfoData["shower"]["general"]),
        len(ds.dfTrainingPfoData["track"]["U"]), len(ds.dfTrainingPfoData["shower"]["U"]),
        len(ds.dfTrainingPfoData["track"]["V"]), len(ds.dfTrainingPfoData["shower"]["V"]),
        len(ds.dfTrainingPfoData["track"]["W"]), len(ds.dfTrainingPfoData["shower"]["W"]),
        len(ds.dfTrainingPfoData["track"]["3D"]), len(ds.dfTrainingPfoData["shower"]["3D"]),
    )
)

def GetViewFeatures(features, view):
    viewFeatures = []
    for feature in features:
        if ds.GetFeatureView(feature['name']) == view:
            viewFeatures.append(feature)
    return viewFeatures


def GetBDTValues(clf, view, valueMask={}):
    if view == "union":
        featureNames = ["BDT3D", "BDTU", "BDTV", "BDTW"]
    else:
        viewFeatures = GetViewFeatures(features, view)
        featureNames = [feature['name'] for feature in viewFeatures]
    trainingDataFeed = ds.dfTrainingPfoData['all'][view]
    for feature in valueMask:
        mask = trainingDataFeed.eval(valueMask[feature])
        trainingDataFeed.loc[mask, feature] = np.nan # Mask feature values using NaNs
    trainingDataFeed = trainingDataFeed[featureNames] # Remove all irrelevant columns
    classificationArray = np.repeat([0,1],[len(ds.dfTrainingPfoData['shower'][view]), len(ds.dfTrainingPfoData['track'][view])])
    clfView = clf.fit(trainingDataFeed, classificationArray)
    return clfView.decision_function(ds.dfAllPfoData[featureNames])

#clf = ensemble.AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(criterion = 'gini', max_depth = 4, min_samples_leaf = 0.05), n_estimators = 1500, learning_rate = 0.5, algorithm = 'SAMME')
#dfPfoData["BDT3D"] = GetBDTValues(clf, '3D')
#dfPfoData["BDTU"] = GetBDTValues(clf, 'U')
#dfPfoData["BDTV"] = GetBDTValues(clf, 'V')
#dfPfoData["BDTW"] = GetBDTValues(clf, 'W')

#dfPfoData.to_pickle(outputPickleFile)

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
ds.dfAllPfoData["BDTMulti"] = GetBDTValues(clf, 'union', valueMask)

ds.SavePickleFile()
print("Finished!")