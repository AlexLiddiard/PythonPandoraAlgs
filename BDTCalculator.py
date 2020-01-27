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

def TrainBDT(clf, featureNames, trainingData, classificationArray):
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

def GetAllBDTData(trainingPfoData, allPfoData, featureViews, class0query):
    bdts = {}
    for view in featureViews:
        print("\nTraining BDT for " + view + " view")
        clf = ensemble.HistGradientBoostingClassifier()
        bdts["BDT" + view] = TrainBDT(clf, featureViews[view], trainingPfoData[view], trainingPfoData[view].eval(class0query))

    dfBdtValues = GetBDTValues(bdts, featureViews, trainingPfoData["union"])
    trainingPfoData["union"] = pd.concat([trainingPfoData["union"], dfBdtValues], axis=1, sort=False)

    print("\nTraining multi-view BDT")
    clf = ensemble.HistGradientBoostingClassifier()
    bdtMulti = TrainBDT(clf, dfBdtValues.columns, trainingPfoData["union"], trainingPfoData["union"].eval(class0query))
    dfBdtValues = GetBDTValues(bdts, featureViews, ds.dfAllPfoData)
    dfBdtValues["BDTMulti"] = bdtMulti.decision_function(dfBdtValues)
    return dfBdtValues