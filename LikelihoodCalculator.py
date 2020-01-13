import numpy as np
import pandas as pd
import math as m
from UpRootFileReader import MicroBooneGeo

myTestArea = "/home/tomalex/Pandora/"
inputPickleFile = myTestArea + '/PythonPandoraAlgs/featureData.bz2'
outputPickleFile = myTestArea + '/PythonPandoraAlgs/featureData.bz2'

trainingFraction = 0.5
trainingPreFilters = {
    "general": (
        'abs(mcPdgCode) != 2112',
        'minCoordX >= @MicroBooneGeo.RangeX[0] + 10',
        'maxCoordX <= @MicroBooneGeo.RangeX[1] - 10',
        'minCoordY >= @MicroBooneGeo.RangeY[0] + 20',
        'maxCoordY <= @MicroBooneGeo.RangeY[1] - 20',
        'minCoordZ >= @MicroBooneGeo.RangeY[0] + 10',
        'maxCoordZ <= @MicroBooneGeo.RangeZ[1] - 10',
    ),
    "U": (
        'purityU>=0.8',
        'completenessU>=0.8',
        'nHitsU>=50',
    ),
    "V": (
        'purityV>=0.8',
        'completenessV>=0.8',
        'nHitsV>=50',
    ),
    "W": (
        'purityW>=0.8',
        'completenessW>=0.8',
        'nHitsW>=50',
    ),
    "3D":
    (
        'purityU>=0.8 and purityV>=0.8 and purityW>=0.8',
        'completenessU>=0.8 and completenessV>=0.8 and completenessW>=0.8',
        'nHits3D >= 50',
    )
}

features = (
    #{'name': 'RSquaredU', 'pdfBins': np.linspace(0, 1, num=50)},
    #{'name': 'RSquaredV', 'pdfBins': np.linspace(0, 1, num=50)},
    #{'name': 'RSquaredW', 'pdfBins': np.linspace(0, 1, num=50)},
    #{'name': 'BinnedHitStdU', 'pdfBins': np.linspace(0, 12, num=50)},
    #{'name': 'BinnedHitStdV', 'pdfBins': np.linspace(0, 12, num=50)},
    #{'name': 'BinnedHitStdW', 'pdfBins': np.linspace(0, 12, num=50)},
    #{'name': 'RadialBinStdU', 'pdfBins': np.linspace(0, 12, num=50)},
    #{'name': 'RadialBinStdV', 'pdfBins': np.linspace(0, 12, num=50)},
    #{'name': 'RadialBinStdW', 'pdfBins': np.linspace(0, 12, num=50),},
    #{'name': 'RadialBinStd3D', 'pdfBins': np.linspace(0, 12, num=50)},
    #{'name': 'ChainCountU', 'pdfBins': np.linspace(1, 50, num=50)},
    #{'name': 'ChainCountV', 'pdfBins': np.linspace(1, 50, num=50)},
    #{'name': 'ChainCountW', 'pdfBins': np.linspace(1, 50, num=50)},
    {'name': 'ChainRatioAvgU', 'pdfBins': np.linspace(0, 1, num=50)},
    {'name': 'ChainRatioAvgV', 'pdfBins': np.linspace(0, 1, num=50)},
    {'name': 'ChainRatioAvgW', 'pdfBins': np.linspace(0, 1, num=50)},
    #{'name': 'ChainRSquaredAvgU', 'pdfBins': np.linspace(0, 1, num=50)},
    #{'name': 'ChainRSquaredAvgV', 'pdfBins': np.linspace(0, 1, num=50)},
    #{'name': 'ChainRSquaredAvgW', 'pdfBins': np.linspace(0, 1, num=50)},
    #{'name': 'ChainRatioStdU', 'pdfBins': np.linspace(0, 0.8, num=50)},
    #{'name': 'ChainRatioStdV', 'pdfBins': np.linspace(0, 0.8, num=50)},
    #{'name': 'ChainRatioStdW', 'pdfBins': np.linspace(0, 0.8, num=50)},
    {'name': 'ChainRSquaredStdU', 'pdfBins': np.linspace(0, 0.8, num=50)},
    {'name': 'ChainRSquaredStdV', 'pdfBins': np.linspace(0, 0.8, num=50)},
    {'name': 'ChainRSquaredStdW', 'pdfBins': np.linspace(0, 0.8, num=50)},
    #{'name': 'AngularSpanU', 'pdfBins': np.linspace(0, m.pi, num=50)},
    #{'name': 'AngularSpanV', 'pdfBins': np.linspace(0, m.pi, num=50)},
    #{'name': 'AngularSpanW', 'pdfBins': np.linspace(0, m.pi, num=50)},
    #{'name': 'AngularSpan3D', 'pdfBins': np.linspace(0, m.pi, num=50)},
    #{'name': 'LongitudinalSpanU', 'pdfBins': np.linspace(0, 400, num=50)},
    #{'name': 'LongitudinalSpanV', 'pdfBins': np.linspace(0, 400, num=50)},
    #{'name': 'LongitudinalSpanW', 'pdfBins': np.linspace(0, 600, num=50)},
    #{'name': 'LongitudinalSpan3D', 'pdfBins': np.linspace(0, 600, num=50)},
    #{'name': 'PcaVarU', 'pdfBins': np.linspace(0, 10, num=50)},
    #{'name': 'PcaVarV', 'pdfBins': np.linspace(0, 10, num=50)},
    #{'name': 'PcaVarW', 'pdfBins': np.linspace(0, 10, num=50)},
    #{'name': 'PcaVar3D', 'pdfBins': np.linspace(0, 10, num=50)},
    #{'name': 'PcaRatioU', 'pdfBins': np.linspace(0, 0.4, num=50)},
    #{'name': 'PcaRatioV', 'pdfBins': np.linspace(0, 0.4, num=50)},
    #{'name': 'PcaRatioW', 'pdfBins': np.linspace(0, 0.4, num=50)},
    {'name': 'PcaRatio3D', 'pdfBins': np.linspace(0, 0.4, num=50)},
    #{'name': 'ChargedBinnedHitStdU', 'pdfBins': np.linspace(0, 100, num=25)},
    #{'name': 'ChargedBinnedHitStdV', 'pdfBins': np.linspace(0, 100, num=25)},
    #{'name': 'ChargedBinnedHitStdW', 'pdfBins': np.linspace(0, 100, num=25)},
    #{'name': 'ChargedStdMeanRatioU', 'pdfBins': np.linspace(0, 4, num=100)},
    #{'name': 'ChargedStdMeanRatioV', 'pdfBins': np.linspace(0, 4, num=100)},
    #{'name': 'ChargedStdMeanRatioW', 'pdfBins': np.linspace(0, 4, num=100)},
    #{'name': 'BraggPeakU', 'pdfBins': np.linspace(0, 1, num=100)},
    #{'name': 'BraggPeakV', 'pdfBins': np.linspace(0, 1, num=100)},
    #{'name': 'BraggPeakW', 'pdfBins': np.linspace(0, 1, num=100)},
    {'name': 'BraggPeak3D', 'pdfBins': np.linspace(0, 1, num=100)},
    {'name': 'Moliere3D', 'pdfBins': np.linspace(0, 0.0002, num=100)}
)

delta = 1e-12

def GetFeatureView(featureName):
    return "3D" if featureName.endswith("3D") else featureName[-1]

def GetViewFeatures(features, view):
    viewFeatures = []
    for feature in features:
        if GetFeatureView(feature['name']) == view:
            viewFeatures.append(feature)
    return viewFeatures

'''Separate true tracks from true showers. Then plot histograms for feature
values. Convert these histograms in to PDFs using scipy. Plot overlapping
histograms for track and shower types.'''
# Load the pickle file.
dfPfoData = pd.read_pickle(inputPickleFile)
nPfoData = len(dfPfoData)

# Get training PFOs.
dfTrainingPfoData = dfPfoData[:m.floor(nPfoData * trainingFraction)]

# Apply training pre-filters.
trainingPreFilters["general"] = ' and '.join(trainingPreFilters["general"])
trainingPreFilters["U"] = ' and '.join(trainingPreFilters["U"])
trainingPreFilters["V"] = ' and '.join(trainingPreFilters["V"])
trainingPreFilters["W"] = ' and '.join(trainingPreFilters["W"])
trainingPreFilters["3D"] = ' and '.join(trainingPreFilters["3D"])

dfTrainingPfoData = {"general": dfTrainingPfoData.query(trainingPreFilters["general"])}
dfTrainingPfoData["shower"] = {"general": dfTrainingPfoData["general"].query("isShower==1")}
dfTrainingPfoData["shower"]["U"] = dfTrainingPfoData["shower"]["general"].query(trainingPreFilters["U"])
dfTrainingPfoData["shower"]["V"] = dfTrainingPfoData["shower"]["general"].query(trainingPreFilters["V"])
dfTrainingPfoData["shower"]["W"] = dfTrainingPfoData["shower"]["general"].query(trainingPreFilters["W"])
dfTrainingPfoData["shower"]["3D"] = dfTrainingPfoData["shower"]["general"].query(trainingPreFilters["3D"])
dfTrainingPfoData["track"] = {"general": dfTrainingPfoData["general"].query("isShower==0")}
dfTrainingPfoData["track"]["U"] = dfTrainingPfoData["track"]["general"].query(trainingPreFilters["U"])
dfTrainingPfoData["track"]["V"] = dfTrainingPfoData["track"]["general"].query(trainingPreFilters["V"])
dfTrainingPfoData["track"]["W"] = dfTrainingPfoData["track"]["general"].query(trainingPreFilters["W"])
dfTrainingPfoData["track"]["3D"] = dfTrainingPfoData["track"]["general"].query(trainingPreFilters["3D"])

viewsUsed = {
    "U": len(GetViewFeatures(features, "U")) > 0,
    "V": len(GetViewFeatures(features, "V")) > 0,
    "W": len(GetViewFeatures(features, "W")) > 0,
    "3D": len(GetViewFeatures(features, "3D")) > 0
}

# Calculate priors
priorViewFilters = []
for key in viewsUsed.keys():
    if viewsUsed[key]:
        priorViewFilters.append(trainingPreFilters[key])
priorFilter = "(" + ") or (".join(priorViewFilters) + ")"
nTracksPrior = len(dfTrainingPfoData["track"]["general"].query(priorFilter))
nShowersPrior = len(dfTrainingPfoData["shower"]["general"].query(priorFilter))

dfPfoData["showerPrior"] = nShowersPrior / (nShowersPrior + nTracksPrior)
dfPfoData["trackPrior"] = nTracksPrior / (nShowersPrior + nTracksPrior)

#Calculate histogram bins, obtain likelihood from them
print((
    "Training likelihood using the following samples:\n" +
    "Priors: %s tracks, %s showers\n" +
    "U View: %s tracks, %s showers\n" +
    "V View: %s tracks, %s showers\n" +
    "W View: %s tracks, %s showers\n" +
    "3D View: %s tracks, %s showers\n") %
    (
        nTracksPrior, nShowersPrior,
        len(dfTrainingPfoData["track"]["U"]), len(dfTrainingPfoData["shower"]["U"]),
        len(dfTrainingPfoData["track"]["V"]), len(dfTrainingPfoData["shower"]["V"]),
        len(dfTrainingPfoData["track"]["W"]), len(dfTrainingPfoData["shower"]["W"]),
        len(dfTrainingPfoData["track"]["3D"]), len(dfTrainingPfoData["shower"]["3D"]),
    )
)
print("Priors: showers %.3f, tracks %.3f" % (dfPfoData["showerPrior"][0], dfPfoData["trackPrior"][0]))

probabilities = {
    "track": {
        "U": np.repeat(1., nPfoData),
        "V": np.repeat(1., nPfoData),
        "W": np.repeat(1., nPfoData),
        "3D": np.repeat(1., nPfoData)
    },
    "shower": {
        "U": np.repeat(1., nPfoData),
        "V": np.repeat(1., nPfoData),
        "W": np.repeat(1., nPfoData),
        "3D": np.repeat(1., nPfoData)
    }
}

for feature in features:
    showerHist, binEdges = np.histogram(dfTrainingPfoData["shower"][GetFeatureView(feature['name'])][feature['name']], bins=feature['pdfBins'], density=True)
    trackHist, binEdges = np.histogram(dfTrainingPfoData["track"][GetFeatureView(feature['name'])][feature['name']], bins=feature['pdfBins'], density=True)
    showerHist = np.concatenate(([1], showerHist, [1])) # values that fall outside the histogram range will not be used for calculating likelihood
    showerHist[showerHist==0] = delta # Avoid nan-valued likelihoods by replacing zero probability densities with a tiny positive number
    trackHist[trackHist==0] = delta
    trackHist = np.concatenate(([1], trackHist, [1]))
    featureValues = dfPfoData[feature['name']]
    histIndices = np.digitize(featureValues, feature['pdfBins'])
    probabilities["track"][GetFeatureView(feature['name'])] *= trackHist[histIndices]
    probabilities["shower"][GetFeatureView(feature['name'])] *= showerHist[histIndices]

if len(GetViewFeatures(features, "U")) > 0:
    dfPfoData["ptU"] = probabilities["track"]["U"]
    dfPfoData["psU"] = probabilities["shower"]["U"]
if len(GetViewFeatures(features, "V")) > 0:
    dfPfoData["ptV"] = probabilities["track"]["V"]
    dfPfoData["psV"] = probabilities["shower"]["V"]
if len(GetViewFeatures(features, "W")) > 0:
    dfPfoData["ptW"] = probabilities["track"]["W"]
    dfPfoData["psW"] = probabilities["shower"]["W"]
if len(GetViewFeatures(features, "3D")) > 0:
    dfPfoData["pt3D"] = probabilities["track"]["3D"]
    dfPfoData["ps3D"] = probabilities["shower"]["3D"]
dfPfoData.to_pickle(outputPickleFile)
print("Finished!")