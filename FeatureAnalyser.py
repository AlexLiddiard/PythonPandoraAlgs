import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import math as m
from PfoGraphicalAnalyser import MicroBooneGeo
from HistoSynthesis import CreateHistogramWire
from LikelihoodAnalyser import GraphCutoffLine, OptimiseCutoff, PrintPurityEfficiency

myTestArea = "/home/tomalex/Pandora"
inputPickleFile = myTestArea + '/PythonPandoraAlgs/featureData.bz2'
trainingFraction = 0.5

preFilters = (
    #'purityU>=0.8',
    #'purityV>=0.8',
    #'purityW>=0.8',
    #'completenessU>=0.8',
    #'completenessV>=0.8',
    #'completenessW>=0.8',
    #'absPdgCode != 2112',
    #'(nHitsU>=10 and nHitsV>=10) or (nHitsU>=10 and nHitsW>=10) or (nHitsV>=10 and nHitsW>=10)',
    #'nHitsU + nHitsV + nHitsW >= 100',
    #'nHitsU>=20',
    #'nHitsV>=20',
    'nHitsW>=20',
    'minCoordX >= @MicroBooneGeo.RangeX[0] + 10',
    'maxCoordX <= @MicroBooneGeo.RangeX[1] - 10',
    'minCoordY >= @MicroBooneGeo.RangeY[0] + 20',
    'maxCoordY <= @MicroBooneGeo.RangeY[1] - 20',
    'minCoordZ >= @MicroBooneGeo.RangeY[0] + 10',
    'maxCoordZ <= @MicroBooneGeo.RangeZ[1] - 10',
)

features = (
    #{'name': 'RSquaredU', 'bins': np.linspace(0, 1, num=50), 'showerCutDirection': 'left'},
    #{'name': 'RSquaredV', 'bins': np.linspace(0, 1, num=50), 'showerCutDirection': 'left'},
    #{'name': 'RSquaredW', 'bins': np.linspace(0, 1, num=50), 'showerCutDirection': 'left'},
    #{'name': 'BinnedHitStdU', 'bins': np.linspace(0, 12, num=50), 'showerCutDirection': 'right'},
    #{'name': 'BinnedHitStdV', 'bins': np.linspace(0, 12, num=50), 'showerCutDirection': 'right'},
    {'name': 'BinnedHitStdW', 'bins': np.linspace(0, 12, num=50), 'showerCutDirection': 'right'},
    #{'name': 'ChainCountU', 'bins': np.linspace(1, 50, num=50), 'showerCutDirection': 'right'},
    #{'name': 'ChainCountV', 'bins': np.linspace(1, 50, num=50), 'showerCutDirection': 'right'},
    #{'name': 'ChainCountW', 'bins': np.linspace(1, 50, num=50), 'showerCutDirection': 'right'},
    #{'name': 'ChainRatioAvgU', 'bins': np.linspace(0, 1, num=50), 'showerCutDirection': 'left'},
    #{'name': 'ChainRatioAvgV', 'bins': np.linspace(0, 1, num=50), 'showerCutDirection': 'left'},
    #{'name': 'ChainRatioAvgW', 'bins': np.linspace(0, 1, num=50), 'showerCutDirection': 'left'},
    #{'name': 'ChainRSquaredAvgU', 'bins': np.linspace(0, 1, num=50), 'showerCutDirection': 'left'},
    #{'name': 'ChainRSquaredAvgV', 'bins': np.linspace(0, 1, num=50), 'showerCutDirection': 'left'},
    #{'name': 'ChainRSquaredAvgW', 'bins': np.linspace(0, 1, num=50), 'showerCutDirection': 'left'},
    #{'name': 'ChainRatioStdU', 'bins': np.linspace(0, 0.8, num=50), 'showerCutDirection': 'right'},
    #{'name': 'ChainRatioStdV', 'bins': np.linspace(0, 0.8, num=50), 'showerCutDirection': 'right'},
    #{'name': 'ChainRatioStdW', 'bins': np.linspace(0, 0.8, num=50), 'showerCutDirection': 'right'},
    #{'name': 'ChainRSquaredStdU', 'bins': np.linspace(0, 0.8, num=50), 'showerCutDirection': 'right'},
    #{'name': 'ChainRSquaredStdV', 'bins': np.linspace(0, 0.8, num=50), 'showerCutDirection': 'right'},
    #{'name': 'ChainRSquaredStdW', 'bins': np.linspace(0, 0.8, num=50), 'showerCutDirection': 'right'},
    #{'name': 'AngularSpanU', 'bins': np.linspace(0, m.pi, num=50), 'showerCutDirection': 'right'},
    #{'name': 'AngularSpanV', 'bins': np.linspace(0, m.pi, num=50), 'showerCutDirection': 'right'},
    #{'name': 'AngularSpanW', 'bins': np.linspace(0, m.pi, num=50), 'showerCutDirection': 'right'},
    #{'name': 'LongitudinalSpanU', 'bins': np.linspace(0, 400, num=50), 'showerCutDirection': 'left'},
    #{'name': 'LongitudinalSpanV', 'bins': np.linspace(0, 400, num=50), 'showerCutDirection': 'left'},
    #{'name': 'LongitudinalSpanW', 'bins': np.linspace(0, 600, num=50), 'showerCutDirection': 'left'},
    #{'name': 'PcaMinVarU', 'bins': np.linspace(0, 10, num=50), 'showerCutDirection': 'right'},
    #{'name': 'PcaMinVarV', 'bins': np.linspace(0, 10, num=50), 'showerCutDirection': 'right'},
    #{'name': 'PcaMinVarW', 'bins': np.linspace(0, 10, num=50), 'showerCutDirection': 'right'},
    #{'name': 'PcaMinRatioU', 'bins': np.linspace(0, 0.4, num=50), 'showerCutDirection': 'right'},
    #{'name': 'PcaMinRatioV', 'bins': np.linspace(0, 0.4, num=50), 'showerCutDirection': 'right'},
    #{'name': 'PcaMinRatioW', 'bins': np.linspace(0, 0.4, num=50), 'showerCutDirection': 'right'},
)

featureHistograms = {
    "plot": True,
    "filters": (
        ("Showers", "isShower==1", "", True), 
        ("Tracks", "isShower==0", "", True),
        ("Electrons + Positrons", "absPdgCode==11", "isShower==1", False),
        ("Photons", "absPdgCode==22",  "isShower==1", False),
        ("Protons", "absPdgCode==2212", "isShower==0", False),
        ("Muons", "absPdgCode==13", "isShower==0", False),
        ("Charged Pions", "absPdgCode==211", "isShower==0", False),
    )   
}
efficiencyPurityPlots = {
    "plot": True,
    "nTestValues": 1000
}

# Load the pickle file.
dfPfoData = pd.read_pickle(inputPickleFile)
# Apply pre-filters.
dfPfoData = dfPfoData.query(' and '.join(preFilters))
dfTrackData = dfPfoData.query("isShower==0")
dfShowerData = dfPfoData.query("isShower==1")
print("Testing feature variables using %d tracks and %d showers." % (len(dfTrackData), len(dfShowerData)))

for feature in features:
    if efficiencyPurityPlots["plot"]:
        # Get optimal purity and efficiency
        testValues = np.linspace(feature["bins"][0], feature["bins"][-1], efficiencyPurityPlots["nTestValues"])
        (
            bestTrackCutoff, trackEfficiencies, trackPurities, trackPurityEfficiencies, 
            bestShowerCutoff, showerEfficiencies, showerPurities, showerPurityEfficiencies
        ) = OptimiseCutoff(dfTrackData, dfShowerData, feature['name'], testValues, feature['showerCutDirection'])

        # Printing results for optimal purity and efficiency
        print("Performance results for %s:" % feature['name'])
        print("\nOptimal track cutoff %.3f" % bestTrackCutoff)
        PrintPurityEfficiency(dfTrackData, dfShowerData, feature['name'], bestTrackCutoff)
        print("\nOptimal shower cutoff %.3f" % bestShowerCutoff)
        PrintPurityEfficiency(dfTrackData, dfShowerData, feature['name'], bestShowerCutoff)
    if featureHistograms["plot"]:
        # Plot histograms
        feature['filters'] = featureHistograms["filters"]
        feature['yAxis'] = 'log'
        fig, ax = CreateHistogramWire(dfPfoData, feature)
        if efficiencyPurityPlots["plot"]:
            GraphCutoffLine(ax, bestShowerCutoff, True, feature['showerCutDirection'] == 'left')
        plt.savefig('%s distribution for %s' % (feature['name'], ', '.join((filter[0] for filter in feature['filters'])) + '.svg'), format='svg', dpi=1200)
        plt.show()
    if efficiencyPurityPlots["plot"]:        
        # Plot purity/efficiency against cutoff
        fig = plt.figure(figsize=(20,7.5))
        bx1 = fig.add_subplot(1,2,1)
        bx2 = fig.add_subplot(1,2,2)
        lines = bx1.plot(testValues, trackPurities, 'b', testValues, trackEfficiencies, 'r', testValues, trackPurityEfficiencies, 'g')
        bx1.legend(lines, ('Purity', 'Efficiency', 'Purity * Efficiency'), loc='lower center')
        lines = bx2.plot(testValues, showerPurities, 'b', testValues, showerEfficiencies, 'r', testValues, showerPurityEfficiencies, 'g')
        bx2.legend(lines, ('Purity', 'Efficiency', 'Purity * Efficiency'), loc='lower center')
        GraphCutoffLine(bx1, bestTrackCutoff)
        GraphCutoffLine(bx2, bestShowerCutoff)

        bx1.set_ylim([0, 1])
        bx1.set_title("Purity/Efficiency vs %s Cutoff — Tracks" % feature['name'])
        bx1.set_xlabel(feature['name'] + " Cutoff")
        bx1.set_ylabel("Fraction")

        bx2.set_ylim([0, 1])
        bx2.set_title("Purity/Efficiency vs %s Cutoff — Showers" % feature['name'])
        bx2.set_xlabel(feature['name'] + " Cutoff")
        bx2.set_ylabel("Fraction")

        plt.tight_layout()
        plt.savefig("PurityEfficiencyVs%sCutoff.svg" % feature['name'], format='svg', dpi=1200)
        plt.show()

featureValuesArray = dfPfoData[(feature['name'] for feature in features)]
rMatrix = featureValuesArray.corr()
rSquaredMatrix = rMatrix * rMatrix

sn.heatmap(rSquaredMatrix, annot=True, annot_kws={"size": 20}, cmap="Blues")
plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
plt.tight_layout()
plt.savefig("FeatureRSquaredMatrix.svg", format='svg', dpi=1200)
plt.show()