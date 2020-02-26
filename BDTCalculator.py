import BaseConfig as bc
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
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import RandomOverSampler as ros
import GeneralConfig as gc
import BDTCalculatorConfig as cfg
import HistoSynthesisConfig as hsc
from OpenPickledFigure import SaveFigure

plt.rcParams.update(hsc.plotStyle)

def TrainBDT(featureNames, trainingData, classificationArray):
    clf = ensemble.HistGradientBoostingClassifier()
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
    for btdName, bdt in bdts.items():
        view = ds.GetFeatureView(btdName)
        btdValues[btdName] = bdt.decision_function(evalData[featureViews[view]])
        results = permutation_importance(bdt, evalData[featureViews[view]], evalData.eval(gc.classQueries[0]))
    return pd.DataFrame(btdValues)

def ShowBDTFeatureImportances(bdts, featureViews, evalData):
    for btdName, bdt in bdts.items():
        view = ds.GetFeatureView(btdName)
        ShowFeatureImportance(bdt, featureViews[view], evalData)

def ShowFeatureImportance(bdt, featureNames, evalData):
    results = permutation_importance(bdt, evalData[featureNames], evalData.eval(gc.classQueries[0]), n_repeats=25)
    perm_sorted_idx = results.importances_mean.argsort()
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.boxplot(results.importances[perm_sorted_idx].T, vert=False, labels=[featureNames[i] for i in perm_sorted_idx], showfliers=False)
    fig.tight_layout()
    SaveFigure(fig, bc.figureFolderFull + '/BDT feature importance - %s.pickle' % ", ".join(featureNames))
    plt.show()

if __name__ == "__main__":
    # Load the training PFOs
    ds.LoadPfoData(cfg.features)

    # U/V/W/3D BTDs
    viewBDTs = {}
    featureViews = ds.GetFeatureViews(cfg.features)
    for view, featureNames in featureViews.items():
        dfPfoData = ds.GetFilteredPfoData("training", "all", "training", view)
        print("\nTraining BDT for " + view + " view")
        viewBDTs["BDT" + view] = TrainBDT(featureNames, dfPfoData, dfPfoData.eval(gc.classQueries[0]))
        if cfg.calculateFeatureImportances:
            print("Calculating feature importance")
            dfPfoData = ds.GetFilteredPfoData("performance", "all", "performance", view)
            ShowFeatureImportance(viewBDTs["BDT" + view], featureNames, dfPfoData)

    # BDTMulti
    dfPfoData = ds.GetFilteredPfoData("training", "all", "training", "union")
    dfViewBDTValues = GetBDTValues(viewBDTs, featureViews, dfPfoData)
    dfPfoData = pd.concat([dfPfoData, dfViewBDTValues], axis=1, sort=False)
    print("\nTraining BDTMulti")
    bdtMulti = TrainBDT(dfViewBDTValues.columns, dfPfoData, dfPfoData.eval(gc.classQueries[0]))
    if cfg.calculateFeatureImportances:
        print("Calculating view importance")
        dfPfoData = ds.GetFilteredPfoData("performance", "all", "performance", "union")
        dfViewBDTValues = GetBDTValues(viewBDTs, featureViews, dfPfoData)
        dfPfoData = pd.concat([dfPfoData, dfViewBDTValues], axis=1, sort=False)
        ShowFeatureImportance(bdtMulti, dfViewBDTValues.columns, dfPfoData)
    # Calculate BDTMulti values
    print("\nCalculating BDTMulti values")
    dfBdtValues = GetBDTValues(viewBDTs, featureViews, ds.dfInputPfoData)
    dfBdtValues["BDTMulti"] = bdtMulti.decision_function(dfBdtValues)

    if cfg.calculateBDTAll:
        print("\nTraining BDTAll")
        # BDTAll
        dfTrainingPfoData = ds.GetFilteredPfoData("training", "all", "training", "union")
        featureNames = ds.GetFeatureNames(cfg.features)
        bdtAll = TrainBDT(featureNames, dfTrainingPfoData, dfTrainingPfoData.eval(gc.classQueries[0]))
        if cfg.calculateFeatureImportances:
            print("Calculating view importance")
            dfPfoData = ds.GetFilteredPfoData("performance", "all", "performance", "union")
            dfPfoData = pd.concat([dfPfoData, dfViewBDTValues], axis=1, sort=False)
            ShowFeatureImportance(bdtAll, featureNames, dfPfoData)
        print("\nCalculating BDTAll values")
        dfBdtValues["BDTAll"] = bdtAll.decision_function(ds.dfInputPfoData[featureNames])

    print("\nSaving results")
    ds.SavePfoData(dfBdtValues, "BDTCalculator")
    print("Finished!")