import numpy as np
import pandas as pd
import math as m
import DataSampler as ds
from UpRootFileReader import MicroBooneGeo

delta = 1e-12

def CalculateLikelihoodValues(dfTrainingClass0, dfTrainingClass1, dfAllPfoData, classNames, features, performancePreFilters):
    probabilities = ({}, {})
    for feature in features:
        featureView = ds.GetFeatureView(feature['name'])
        class0Hist, binEdges = np.histogram(dfTrainingClass0[featureView][feature['name']], bins=feature['pdfBins'], density=True)
        class1Hist, binEdges = np.histogram(dfTrainingClass1[featureView][feature['name']], bins=feature['pdfBins'], density=True)
        class0Hist = np.concatenate(([1], class0Hist, [1])) # values that fall outside the histogram range will not be used for calculating likelihood
        class1Hist = np.concatenate(([1], class1Hist, [1]))
        class0Hist[class0Hist==0] = delta # Avoid nan-valued likelihoods by replacing zero probability densities with a tiny positive number
        class1Hist[class1Hist==0] = delta
        featureValues = dfAllPfoData[feature['name']]
        histIndices = np.digitize(featureValues, feature['pdfBins'])
        if featureView not in probabilities[0]:
            probabilities[0][featureView] = 1
            probabilities[1][featureView] = 1
        probabilities[0][featureView] *= class0Hist[histIndices]
        probabilities[1][featureView] *= class1Hist[histIndices]

    # Obtain likelihood values, using only probabilities from views that the PFO satisfied the filters.
    nClass0 = len(dfTrainingClass0["union"])
    nClass1 = len(dfTrainingClass1["union"])
    class0Prior = nClass0 / (nClass0 + nClass1)
    class1Prior = 1 - class0Prior
    p0 = np.repeat(class0Prior, len(dfAllPfoData))
    p1 = np.repeat(class1Prior, len(dfAllPfoData))
    for key in ds.GetViewsUsed(features):
        pfoCheck = dfAllPfoData.eval(performancePreFilters[key])
        p0 *= (1 + pfoCheck * (probabilities[0][key] - 1))
        p1 *= (1 + pfoCheck * (probabilities[1][key] - 1))
    return p1 / (p1 + p0)