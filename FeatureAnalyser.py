import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from PfoGraphAnalysis import MicroBooneGeo
from HistoSynthesis import CreateHistogramWire


myTestArea = "/home/alexliddiard/Desktop/Pandora"
inputPickleFile = myTestArea + '/PythonPandoraAlgs/featureData.bz2'
trainingFraction = 0.5
showHistograms = True
features = ({'name': 'F0aU', 'bins': np.linspace(0, 1, num=50), 'yAxis': 'log'},
            {'name': 'F0aV', 'bins': np.linspace(0, 1, num=50), 'yAxis': 'log'},
            {'name': 'F0aW', 'bins': np.linspace(0, 1, num=50), 'yAxis': 'log'},
            {'name': 'F1aU', 'bins': np.linspace(0, 6, num=50), 'yAxis': 'log'},
            {'name': 'F1aV', 'bins': np.linspace(0, 6, num=50), 'yAxis': 'log'},
            {'name': 'F1aW', 'bins': np.linspace(0, 6, num=50), 'yAxis': 'log'},
            {'name': 'F2aU', 'bins': np.linspace(1, 50, num=50), 'yAxis': 'log'},
            {'name': 'F2aV', 'bins': np.linspace(1, 50, num=50), 'yAxis': 'log'},
            {'name': 'F2aW', 'bins': np.linspace(1, 50, num=50), 'yAxis': 'log'},
            {'name': 'F2bU', 'bins': np.linspace(0, 1, num=50), 'yAxis': 'log'},
            {'name': 'F2bV', 'bins': np.linspace(0, 1, num=50), 'yAxis': 'log'},
            {'name': 'F2bW', 'bins': np.linspace(0, 1, num=50), 'yAxis': 'log'},
            {'name': 'F2cU', 'bins': np.linspace(0, 1, num=50), 'yAxis': 'log'},
            {'name': 'F2cV', 'bins': np.linspace(0, 1, num=50), 'yAxis': 'log'},
            {'name': 'F2cW', 'bins': np.linspace(0, 1, num=50), 'yAxis': 'log'},
            {'name': 'F2dU', 'bins': np.linspace(0, 1, num=50), 'yAxis': 'log'},
            {'name': 'F2dV', 'bins': np.linspace(0, 1, num=50), 'yAxis': 'log'},
            {'name': 'F2dW', 'bins': np.linspace(0, 1, num=50), 'yAxis': 'log'},
            {'name': 'F2eU', 'bins': np.linspace(0, 1, num=50), 'yAxis': 'log'},
            {'name': 'F2eV', 'bins': np.linspace(0, 1, num=50), 'yAxis': 'log'},
            {'name': 'F2eW', 'bins': np.linspace(0, 1, num=50), 'yAxis': 'log'},
            {'name': 'F3aU', 'bins': np.linspace(0, 1.57, num=50), 'yAxis': 'log'},
            {'name': 'F3aV', 'bins': np.linspace(0, 1.57, num=50), 'yAxis': 'log'},
            {'name': 'F3aW', 'bins': np.linspace(0, 1.57, num=50), 'yAxis': 'log'},
            {'name': 'F3bU', 'bins': np.linspace(0, 600, num=100), 'yAxis': 'log'},
            {'name': 'F3bV', 'bins': np.linspace(0, 600, num=100), 'yAxis': 'log'},
            {'name': 'F3bW', 'bins': np.linspace(0, 600, num=100), 'yAxis': 'log'}
)

featureHistFilters = (("Showers", "isShower==1", "", True), 
                      ("Tracks", "isShower==0", "", True),
                      ("Electrons/Positrons", "absPdgCode==11", "isShower==1", False),
                      ("Photons", "absPdgCode==22",  "isShower==1", False),
                      ("Protons", "absPdgCode==2212", "isShower==0", False),
                      ("Muons", "absPdgCode==13", "isShower==0", False),
                      ("Charged Pions", "absPdgCode==211", "isShower==0", False),
)

preFilters = ('purityU>=0.8',
              'purityV>=0.8',
              'purityW>=0.8',
              'completenessU>=0.8',
              'completenessV>=0.8',
              'completenessW>=0.8',
              #'(nHitsU>=10 and nHitsV>=10) or (nHitsU>=10 and nHitsW>=10) or (nHitsV>=10 and nHitsW>=10)',
              #'nHitsU + nHitsV + nHitsW >= 100',
              'nHitsU>=50',
              'nHitsV>=50',
              'nHitsW>=50',
              'absPdgCode != 2112',
              'minCoordX >= @MicroBooneGeo.RangeX[0] + 10',
              'maxCoordX <= @MicroBooneGeo.RangeX[1] - 10',
              'minCoordY >= @MicroBooneGeo.RangeY[0] + 20',
              'maxCoordY <= @MicroBooneGeo.RangeY[1] - 20',
              'minCoordZ >= @MicroBooneGeo.RangeY[0] + 10',
              'maxCoordZ <= @MicroBooneGeo.RangeZ[1] - 10',
)

# Load the pickle file.
dfPfoData = pd.read_pickle(inputPickleFile)
# Apply pre-filters.
dfPfoData = dfPfoData.query(' and '.join(preFilters))

# Make feature histograms.
if showHistograms:
    for feature in features:
        feature['filters'] = featureHistFilters
        CreateHistogramWire(dfPfoData, feature)

featureValuesArray = dfPfoData[(feature['name'] for feature in features)]
corrMatrix = abs(featureValuesArray.corr())
print(corrMatrix)

sn.heatmap(corrMatrix, annot=True, cmap="Blues")
plt.show()