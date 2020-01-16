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
    {'name': 'RSquaredU', 'pdfBins': np.linspace(0, 1, num=50)},
    {'name': 'RSquaredV', 'pdfBins': np.linspace(0, 1, num=50)},
    {'name': 'RSquaredW', 'pdfBins': np.linspace(0, 1, num=50)},
    {'name': 'BinnedHitStdU', 'pdfBins': np.linspace(0, 12, num=50)},
    {'name': 'BinnedHitStdV', 'pdfBins': np.linspace(0, 12, num=50)},
    {'name': 'BinnedHitStdW', 'pdfBins': np.linspace(0, 12, num=50)},
    {'name': 'RadialBinStdU', 'pdfBins': np.linspace(0, 12, num=50)},
    {'name': 'RadialBinStdV', 'pdfBins': np.linspace(0, 12, num=50)},
    {'name': 'RadialBinStdW', 'pdfBins': np.linspace(0, 12, num=50),},
    {'name': 'RadialBinStd3D', 'pdfBins': np.linspace(0, 12, num=50)},
    {'name': 'ChainCountU', 'pdfBins': np.linspace(1, 50, num=50)},
    {'name': 'ChainCountV', 'pdfBins': np.linspace(1, 50, num=50)},
    {'name': 'ChainCountW', 'pdfBins': np.linspace(1, 50, num=50)},
    {'name': 'ChainRatioAvgU', 'pdfBins': np.linspace(0, 1, num=50)},
    {'name': 'ChainRatioAvgV', 'pdfBins': np.linspace(0, 1, num=50)},
    {'name': 'ChainRatioAvgW', 'pdfBins': np.linspace(0, 1, num=50)},
    {'name': 'ChainRSquaredAvgU', 'pdfBins': np.linspace(0, 1, num=50)},
    {'name': 'ChainRSquaredAvgV', 'pdfBins': np.linspace(0, 1, num=50)},
    {'name': 'ChainRSquaredAvgW', 'pdfBins': np.linspace(0, 1, num=50)},
    {'name': 'ChainRatioStdU', 'pdfBins': np.linspace(0, 0.8, num=50)},
    {'name': 'ChainRatioStdV', 'pdfBins': np.linspace(0, 0.8, num=50)},
    {'name': 'ChainRatioStdW', 'pdfBins': np.linspace(0, 0.8, num=50)},
    {'name': 'ChainRSquaredStdU', 'pdfBins': np.linspace(0, 0.8, num=50)},
    {'name': 'ChainRSquaredStdV', 'pdfBins': np.linspace(0, 0.8, num=50)},
    {'name': 'ChainRSquaredStdW', 'pdfBins': np.linspace(0, 0.8, num=50)},
    {'name': 'AngularSpanU', 'pdfBins': np.linspace(0, m.pi, num=50)},
    {'name': 'AngularSpanV', 'pdfBins': np.linspace(0, m.pi, num=50)},
    {'name': 'AngularSpanW', 'pdfBins': np.linspace(0, m.pi, num=50)},
    {'name': 'AngularSpan3D', 'pdfBins': np.linspace(0, m.pi, num=50)},
    {'name': 'LongitudinalSpanU', 'pdfBins': np.linspace(0, 400, num=50)},
    {'name': 'LongitudinalSpanV', 'pdfBins': np.linspace(0, 400, num=50)},
    {'name': 'LongitudinalSpanW', 'pdfBins': np.linspace(0, 600, num=50)},
    {'name': 'LongitudinalSpan3D', 'pdfBins': np.linspace(0, 600, num=50)},
    {'name': 'PcaVarU', 'pdfBins': np.linspace(0, 10, num=50)},
    {'name': 'PcaVarV', 'pdfBins': np.linspace(0, 10, num=50)},
    {'name': 'PcaVarW', 'pdfBins': np.linspace(0, 10, num=50)},
    {'name': 'PcaVar3D', 'pdfBins': np.linspace(0, 10, num=50)},
    {'name': 'PcaRatioU', 'pdfBins': np.linspace(0, 0.4, num=50)},
    {'name': 'PcaRatioV', 'pdfBins': np.linspace(0, 0.4, num=50)},
    {'name': 'PcaRatioW', 'pdfBins': np.linspace(0, 0.4, num=50)},
    {'name': 'PcaRatio3D', 'pdfBins': np.linspace(0, 0.4, num=50)},
    {'name': 'ChargedBinnedHitStdU', 'pdfBins': np.linspace(0, 100, num=25)},
    {'name': 'ChargedBinnedHitStdV', 'pdfBins': np.linspace(0, 100, num=25)},
    {'name': 'ChargedBinnedHitStdW', 'pdfBins': np.linspace(0, 100, num=25)},
    {'name': 'ChargedStdMeanRatioU', 'pdfBins': np.linspace(0, 4, num=100)},
    {'name': 'ChargedStdMeanRatioV', 'pdfBins': np.linspace(0, 4, num=100)},
    {'name': 'ChargedStdMeanRatioW', 'pdfBins': np.linspace(0, 4, num=100)},
    {'name': 'BraggPeakU', 'pdfBins': np.linspace(0, 1, num=100)},
    {'name': 'BraggPeakV', 'pdfBins': np.linspace(0, 1, num=100)},
    {'name': 'BraggPeakW', 'pdfBins': np.linspace(0, 1, num=100)},
    {'name': 'BraggPeak3D', 'pdfBins': np.linspace(0, 1, num=100)},
    {'name': 'Moliere3D', 'pdfBins': np.linspace(0, 0.0002, num=100)}
)

# Load the training PFOs
ds.GetTrainingPfoData()

print((
    "Training likelihood using the following samples:\n" +
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


def GetBDTValues(clf, view):
    if view == "union":
        featureNames = ["BDT3D", "BDTU", "BDTV", "BDTW"]
    else:
        viewFeatures = GetViewFeatures(features, view)
        featureNames = [feature['name'] for feature in viewFeatures]
    trainingDataFeed = pd.concat((ds.dfTrainingPfoData['shower'][view], ds.dfTrainingPfoData['track'][view]))[featureNames]
    numberOfShowerPfos = len(ds.dfTrainingPfoData['shower'][view])
    numberOfTrackPfos = len(ds.dfTrainingPfoData['track'][view])
    classificationArray = np.repeat([0,1],[numberOfShowerPfos, numberOfTrackPfos])
    clfView = clf.fit(trainingDataFeed, classificationArray)
    return clfView.decision_function(ds.dfPfoData[featureNames])

#clf = ensemble.AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(criterion = 'gini', max_depth = 4, min_samples_leaf = 0.05), n_estimators = 1500, learning_rate = 0.5, algorithm = 'SAMME')
#dfPfoData["BDT3D"] = GetBDTValues(clf, '3D')
#dfPfoData["BDTU"] = GetBDTValues(clf, 'U')
#dfPfoData["BDTV"] = GetBDTValues(clf, 'V')
#dfPfoData["BDTW"] = GetBDTValues(clf, 'W')

#dfPfoData.to_pickle(outputPickleFile)

print("Calculating BDT values")
clf = ensemble.HistGradientBoostingClassifier()
ds.dfPfoData["BDT3D"] = GetBDTValues(clf, '3D')
ds.dfPfoData["BDTU"] = GetBDTValues(clf, 'U')
ds.dfPfoData["BDTV"] = GetBDTValues(clf, 'V')
ds.dfPfoData["BDTW"] = GetBDTValues(clf, 'W')
ds.GetTrainingPfoData()
ds.dfPfoData["BDTMulti"] = GetBDTValues(clf, 'union')

ds.SavePickleFile()
print("Finished!")