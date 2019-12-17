import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import math as m
from PfoGraphicalAnalyser import MicroBooneGeo
from HistoSynthesis import CreateHistogramWire


myTestArea = "/home/tomalex/Pandora"
inputPickleFile = myTestArea + '/PythonPandoraAlgs/featureData.bz2'
trainingFraction = 0.5

preFilters = (
    #'purityU>=0.8',
    #'purityV>=0.8',
    'purityW>=0.8',
    #'completenessU>=0.8',
    #'completenessV>=0.8',
    'completenessW>=0.8',
    #'absPdgCode != 2112',
    #'(nHitsU>=10 and nHitsV>=10) or (nHitsU>=10 and nHitsW>=10) or (nHitsV>=10 and nHitsW>=10)',
    #'nHitsU + nHitsV + nHitsW >= 100',
    #'nHitsU>=20',
    #'nHitsV>=20',
    'nHitsW>=50',
    'minCoordX >= @MicroBooneGeo.RangeX[0] + 10',
    'maxCoordX <= @MicroBooneGeo.RangeX[1] - 10',
    'minCoordY >= @MicroBooneGeo.RangeY[0] + 20',
    'maxCoordY <= @MicroBooneGeo.RangeY[1] - 20',
    'minCoordZ >= @MicroBooneGeo.RangeY[0] + 10',
    'maxCoordZ <= @MicroBooneGeo.RangeZ[1] - 10',
)

features = (
    #{'name': 'RSquaredU', 'bins': np.linspace(0, 1, num=50)},
    #{'name': 'RSquaredV', 'bins': np.linspace(0, 1, num=50)},
    {'name': 'RSquaredW', 'bins': np.linspace(0, 1, num=50)},
    #{'name': 'BinnedHitStdU', 'bins': np.linspace(0, 12, num=50)},
    #{'name': 'BinnedHitStdV', 'bins': np.linspace(0, 12, num=50)},
    {'name': 'BinnedHitStdW', 'bins': np.linspace(0, 12, num=50)},
    #{'name': 'ChainCountU', 'bins': np.linspace(1, 50, num=50)},
    #{'name': 'ChainCountV', 'bins': np.linspace(1, 50, num=50)},
    #{'name': 'ChainCountW', 'bins': np.linspace(1, 50, num=50)},
    #{'name': 'ChainRatioAvgU', 'bins': np.linspace(0, 1, num=50)},
    #{'name': 'ChainRatioAvgV', 'bins': np.linspace(0, 1, num=50)},
    {'name': 'ChainRatioAvgW', 'bins': np.linspace(0, 1, num=50)},
    #{'name': 'ChainRSquaredAvgU', 'bins': np.linspace(0, 1, num=50)},
    #{'name': 'ChainRSquaredAvgV', 'bins': np.linspace(0, 1, num=50)},
    #{'name': 'ChainRSquaredAvgW', 'bins': np.linspace(0, 1, num=50)},
    #{'name': 'ChainRatioStdU', 'bins': np.linspace(0, 0.8, num=50)},
    #{'name': 'ChainRatioStdV', 'bins': np.linspace(0, 0.8, num=50)},
    #{'name': 'ChainRatioStdW', 'bins': np.linspace(0, 0.8, num=50)},
    #{'name': 'ChainRSquaredStdU', 'bins': np.linspace(0, 0.8, num=50)},
    #{'name': 'ChainRSquaredStdV', 'bins': np.linspace(0, 0.8, num=50)},
    {'name': 'ChainRSquaredStdW', 'bins': np.linspace(0, 0.8, num=50)},
    #{'name': 'AngularSpanU', 'bins': np.linspace(0, m.pi, num=50)},
    #{'name': 'AngularSpanV', 'bins': np.linspace(0, m.pi, num=50)},
    {'name': 'AngularSpanW', 'bins': np.linspace(0, m.pi, num=50)},
    #{'name': 'LongitudinalSpanU', 'bins': np.linspace(0, 400, num=50)},
    #{'name': 'LongitudinalSpanV', 'bins': np.linspace(0, 400, num=50)},
    {'name': 'LongitudinalSpanW', 'bins': np.linspace(0, 600, num=50)},
)

showHistograms = True
featureHistFilters = (
    ("Showers", "isShower==1", "", True), 
    ("Tracks", "isShower==0", "", True),
    ("Electrons + Positrons", "absPdgCode==11", "isShower==1", False),
    ("Photons", "absPdgCode==22",  "isShower==1", False),
    ("Protons", "absPdgCode==2212", "isShower==0", False),
    ("Muons", "absPdgCode==13", "isShower==0", False),
    ("Charged Pions", "absPdgCode==211", "isShower==0", False),
)

# Load the pickle file.
dfPfoData = pd.read_pickle(inputPickleFile)
# Apply pre-filters.
if preFilters:
    dfPfoData = dfPfoData.query(' and '.join(preFilters))

# Make feature histograms.
if showHistograms:
    for feature in features:
        feature['filters'] = featureHistFilters
        feature['yAxis'] = 'log'
        CreateHistogramWire(dfPfoData, feature)

featureValuesArray = dfPfoData[(feature['name'] for feature in features)]
rMatrix = featureValuesArray.corr()
rSquaredMatrix = rMatrix * rMatrix

sn.heatmap(rSquaredMatrix, annot=True, annot_kws={"size": 15}, cmap="Blues")
plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
plt.tight_layout()
plt.savefig("FeatureRSquaredMatrix.svg", format='svg', dpi=1200)
plt.show()