import numpy as np
import pandas as pd
import math as m
import BaseConfig as bc
import DataSampler as ds
import LikelihoodCalculatorConfig as cfg
import GeneralConfig as gc
import DataSamplerConfig as dsc
from UpRootFileReader import MicroBooneGeo

delta = 1e-12

def CalculateLikelihoodValues(features):
    probabilities = ({}, {})
    featureViews = ds.GetFeatureViews(features)
    for featureView, featureNames in featureViews.items():
        # Load class0 and class1 data sample for view
        dfClass0 = ds.GetFilteredPfoData("performance", gc.classNames[0], "performance", featureView)
        dfClass1 = ds.GetFilteredPfoData("performance", gc.classNames[1], "performance", featureView)
        for featureName in featureNames:
            pdfBins = [feature for feature in features if feature["name"] == featureName][0]["pdfBins"]
            class0Hist, binEdges = np.histogram(dfClass0[featureName], bins=pdfBins, density=True)
            class1Hist, binEdges = np.histogram(dfClass1[featureName], bins=pdfBins, density=True)
            class0Hist = np.concatenate(([1], class0Hist, [1])) # values that fall outside the histogram range will not be used for calculating likelihood
            class1Hist = np.concatenate(([1], class1Hist, [1]))
            class0Hist[class0Hist==0] = delta # Avoid nan-valued likelihoods by replacing zero probability densities with a tiny positive number
            class1Hist[class1Hist==0] = delta
            featureValues = ds.dfInputPfoData[featureName]
            histIndices = np.digitize(featureValues, pdfBins)
            if featureView not in probabilities[0]:
                probabilities[0][featureView] = 1
                probabilities[1][featureView] = 1
            probabilities[0][featureView] *= class0Hist[histIndices]
            probabilities[1][featureView] *= class1Hist[histIndices]

    # Obtain likelihood values, using only probabilities from views that the PFO satisfied the filters.
    nClass0 = len(ds.GetFilteredPfoData("performance", gc.classNames[0], "performance", "union"))
    nClass1 = len(ds.GetFilteredPfoData("performance", gc.classNames[1], "performance", "union"))
    class0Prior = nClass0 / (nClass0 + nClass1)
    class1Prior = 1 - class0Prior
    p0 = np.repeat(class0Prior, len(ds.dfInputPfoData))
    p1 = np.repeat(class1Prior, len(ds.dfInputPfoData))

    for view in ds.GetFeatureViews(features):
        pfoCheck = ds.dfInputPfoData.eval(dsc.preFilters["performance"][view])
        p0 *= (1 + pfoCheck * (probabilities[0][view] - 1))
        p1 *= (1 + pfoCheck * (probabilities[1][view] - 1))
    return p1 / (p1 + p0)

if __name__ == "__main__":
    ds.LoadPfoData(cfg.features)
    # Calculate likelihood values
    print("Calculating likelihood values")
    likelihoodValues = CalculateLikelihoodValues(cfg.features)
    dfLikelihood = pd.DataFrame({"Likelihood": likelihoodValues})

    # Save the results
    ds.SavePfoData(dfLikelihood, "LikelihoodCalculator")
    print("Finished!")