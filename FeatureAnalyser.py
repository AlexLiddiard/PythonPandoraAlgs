import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from HistoSynthesis import CreateHistogram
import seaborn as sn

myTestArea = "/home/alexliddiard/Desktop/Pandora"
inputPickleFile = myTestArea + '/PythonPandoraAlgs/featureDataTemp.pickle'
trainingFraction = 0.5
showHistograms = False
features = ({'name': 'F0aU', 'bins': np.linspace(0, 1, num=50)},
            {'name': 'F0aV', 'bins': np.linspace(0, 1, num=50)},
            {'name': 'F0aW', 'bins': np.linspace(0, 1, num=50)},
            {'name': 'F1aU', 'bins': np.linspace(0, 6, num=50)},
            {'name': 'F1aV', 'bins': np.linspace(0, 6, num=50)},
            {'name': 'F1aW', 'bins': np.linspace(0, 6, num=50)},
            {'name': 'F2aU', 'bins': np.linspace(0, 30, num=31)},
            {'name': 'F2aV', 'bins': np.linspace(0, 30, num=31)},
            {'name': 'F2aW', 'bins': np.linspace(0, 30, num=31)},
            {'name': 'F2bU', 'bins': np.linspace(0, 1, num=50)},
            {'name': 'F2bV', 'bins': np.linspace(0, 1, num=50)},
            {'name': 'F2bW', 'bins': np.linspace(0, 1, num=50)},
            {'name': 'F2cU', 'bins': np.linspace(0, 1, num=50)},
            {'name': 'F2cV', 'bins': np.linspace(0, 1, num=50)},
            {'name': 'F2cW', 'bins': np.linspace(0, 1, num=50)},
            {'name': 'F2dU', 'bins': np.linspace(0, 1, num=50)},
            {'name': 'F2dV', 'bins': np.linspace(0, 1, num=50)},
            {'name': 'F2dW', 'bins': np.linspace(0, 1, num=50)},
            {'name': 'F2eU', 'bins': np.linspace(0, 1, num=50)},
            {'name': 'F2eV', 'bins': np.linspace(0, 1, num=50)},
            {'name': 'F2eW', 'bins': np.linspace(0, 1, num=50)},
            {'name': 'F3aU', 'bins': np.linspace(0, 1.57, num=50)},
            {'name': 'F3aV', 'bins': np.linspace(0, 1.57, num=50)},
            {'name': 'F3aW', 'bins': np.linspace(0, 1.57, num=50)},
            {'name': 'F3bU', 'bins': np.linspace(0, 1000, num=100)},
            {'name': 'F3bV', 'bins': np.linspace(0, 1000, num=100)},
            {'name': 'F3bW', 'bins': np.linspace(0, 1000, num=100)}
           )
preFilters = ('purityU>=0.8',
              'purityV>=0.8',
              'purityW>=0.8',
              'completenessU>=0.8',
              'completenessV>=0.8',
              'completenessW>=0.8',
              'nHitsU>=10',
              'nHitsV>=10',
              'nHitsW>=10',
              'absPdgCode not in [2112, 14, 12]')

# Load the pickle file.
dfPfoData = pd.read_pickle(inputPickleFile)
# Apply pre-filters.
dfPfoData = dfPfoData.query(' and '.join(preFilters))

# Make feature histograms.
if showHistograms:
    for feature in features:
        feature['filters'] = [("isShower==1", "Showers"), ("isShower==0", "Tracks")]
        CreateHistogram(dfPfoData, feature)

featureValuesArray = dfPfoData[(feature['name'] for feature in features)]
corrMatrix = abs(featureValuesArray.corr())
print(corrMatrix)

sn.heatmap(corrMatrix, annot=True, cmap="Blues")
plt.show()