import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import math as m
from UpRootFileReader import MicroBooneGeo
from HistoSynthesis import CreateHistogramWire
from LikelihoodAnalyser import GraphCutoffLine, OptimiseCutoff, PrintPurityEfficiency

myTestArea = "/home/tomalex/Pandora/"
inputPickleFile = myTestArea + '/PythonPandoraAlgs/featureData.bz2'
trainingFraction = 0.5
preFilters = {
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
        #'purityU>=0.8',
        #'completenessU>=0.8',
        'nHitsU>=20',
    ),
    "V": (
        #'purityV>=0.8',
        #'completenessV>=0.8',
        'nHitsV>=20',
    ),
    "W": (
        #'purityW>=0.8',
        #'completenessW>=0.8',
        'nHitsW>=20',
    ),
    "3D":
    (
        #'purityU>=0.8 and purityV>=0.8 and purityW>=0.8',
        #'completenessU>=0.8 and completenessV>=0.8 and completenessW>=0.8',
        'nHits3D>=20',
    )
}

features = (
    {'name': 'RSquaredU', 'bins': np.linspace(0, 1, num=50), 'showerCutDirection': 'left'},
    #{'name': 'RSquaredV', 'bins': np.linspace(0, 1, num=50), 'showerCutDirection': 'left'},
    #{'name': 'RSquaredW', 'bins': np.linspace(0, 1, num=50), 'showerCutDirection': 'left'},
    #{'name': 'BinnedHitStdU', 'bins': np.linspace(0, 12, num=50), 'showerCutDirection': 'right'},
    #{'name': 'BinnedHitStdV', 'bins': np.linspace(0, 12, num=50), 'showerCutDirection': 'right'},
    #{'name': 'BinnedHitStdW', 'bins': np.linspace(0, 12, num=50), 'showerCutDirection': 'right'},
    #{'name': 'RadialBinStdU', 'bins': np.linspace(0, 12, num=50), 'showerCutDirection': 'right'},
    #{'name': 'RadialBinStdV', 'bins': np.linspace(0, 12, num=50), 'showerCutDirection': 'right'},
    #{'name': 'RadialBinStdW', 'bins': np.linspace(0, 12, num=50), 'showerCutDirection': 'right'},
    #{'name': 'RadialBinStd3D', 'bins': np.linspace(0, 12, num=50), 'showerCutDirection': 'right'},
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
    #{'name': 'AngularSpan3D', 'bins': np.linspace(0, m.pi, num=50), 'showerCutDirection': 'right'},
    #{'name': 'LongitudinalSpanU', 'bins': np.linspace(0, 400, num=50), 'showerCutDirection': 'left'},
    #{'name': 'LongitudinalSpanV', 'bins': np.linspace(0, 600, num=50), 'showerCutDirection': 'left'},
    #{'name': 'LongitudinalSpanW', 'bins': np.linspace(0, 600, num=50), 'showerCutDirection': 'left'},
    #{'name': 'LongitudinalSpan3D', 'bins': np.linspace(0, 600, num=50), 'showerCutDirection': 'left'},
    #{'name': 'PcaVarU', 'bins': np.linspace(0, 10, num=50), 'showerCutDirection': 'right'},
    #{'name': 'PcaVarV', 'bins': np.linspace(0, 10, num=50), 'showerCutDirection': 'right'},
    #{'name': 'PcaVarW', 'bins': np.linspace(0, 10, num=50), 'showerCutDirection': 'right'},
    #{'name': 'PcaVar3D', 'bins': np.linspace(0, 10, num=50), 'showerCutDirection': 'right'},
    #{'name': 'PcaRatioU', 'bins': np.linspace(0, 0.4, num=50), 'showerCutDirection': 'right'},
    #{'name': 'PcaRatioV', 'bins': np.linspace(0, 0.4, num=50), 'showerCutDirection': 'right'},
    #{'name': 'PcaRatioW', 'bins': np.linspace(0, 0.4, num=50), 'showerCutDirection': 'right'},
    #{'name': 'PcaRatio3D', 'bins': np.linspace(0, 1, num=50), 'showerCutDirection': 'right'},
    #{'name': 'ChargedBinnedHitStdU', 'bins': np.linspace(0, 100, num=25), 'showerCutDirection': 'right'},
    #{'name': 'ChargedBinnedHitStdV', 'bins': np.linspace(0, 100, num=25), 'showerCutDirection': 'right'},
    #{'name': 'ChargedBinnedHitStdW', 'bins': np.linspace(0, 100, num=25), 'showerCutDirection': 'right'},
    #{'name': 'ChargedStdMeanRatioU', 'bins': np.linspace(0, 4, num=100), 'showerCutDirection': 'right'},
    #{'name': 'ChargedStdMeanRatioV', 'bins': np.linspace(0, 4, num=100), 'showerCutDirection': 'right'},
    #{'name': 'ChargedStdMeanRatioW', 'bins': np.linspace(0, 4, num=100), 'showerCutDirection': 'right'},
    #{'name': 'ChargedStdMeanRatio3D', 'bins': np.linspace(0, 4, num=100), 'showerCutDirection': 'right'},
    #{'name': 'BraggPeakU', 'bins': np.linspace(0, 1, num=100), 'showerCutDirection': 'left'},
    #{'name': 'BraggPeakV', 'bins': np.linspace(0, 1, num=100), 'showerCutDirection': 'left'},
    #{'name': 'BraggPeakW', 'bins': np.linspace(0, 1, num=100), 'showerCutDirection': 'left'},
    #{'name': 'BraggPeak3D', 'bins': np.linspace(0, 1, num=100), 'showerCutDirection': 'left'},
    #{'name': 'Moliere3D', 'bins': np.linspace(0, 0.0002, num=100), 'showerCutDirection': 'right'},
    #{'name': 'BDTU', 'bins': np.linspace(-0.5, 0.5, num = 40), 'showerCutDirection': 'left'},
    #{'name': 'BDTV', 'bins': np.linspace(-0.5, 0.5, num = 40), 'showerCutDirection': 'left'},
    #{'name': 'BDTW', 'bins': np.linspace(-0.5, 0.5, num = 40), 'showerCutDirection': 'left'},
    #{'name': 'BDT3D', 'bins': np.linspace(-0.5, 0.5, num = 40), 'showerCutDirection': 'left'},
)

featureHistograms = {
    "plot": True,
    "filters": (
        ("Showers", "isShower==1", "", True), 
        ("Tracks", "isShower==0", "", True),
        ("Electrons + Positrons", "abs(mcPdgCode)==11", "isShower==1", False),
        ("Photons", "abs(mcPdgCode)==22",  "isShower==1", False),
        ("Protons", "abs(mcPdgCode)==2212", "isShower==0", False),
        ("Muons", "abs(mcPdgCode)==13", "isShower==0", False),
        ("Charged Pions", "abs(mcPdgCode)==211", "isShower==0", False),
    )   
}
efficiencyPurityPlots = {
    "plot": True,
    "nTestValues": 1000
}

def GetFeatureView(featureName):
    return "3D" if featureName.endswith("3D") else featureName[-1]

# Load the pickle file.
dfPfoData = pd.read_pickle(inputPickleFile)
# Apply pre-filters.
preFilters["general"] = ' and '.join(preFilters["general"])
preFilters["U"] = ' and '.join(preFilters["U"])
preFilters["V"] = ' and '.join(preFilters["V"])
preFilters["W"] = ' and '.join(preFilters["W"])
preFilters["3D"] = ' and '.join(preFilters["3D"])

dfPfoData = {"general": dfPfoData.query(preFilters["general"])}
dfPfoData["shower"] = {"general": dfPfoData["general"].query("isShower==1")}
dfPfoData["shower"]["U"] = dfPfoData["shower"]["general"].query(preFilters["U"])
dfPfoData["shower"]["V"] = dfPfoData["shower"]["general"].query(preFilters["V"])
dfPfoData["shower"]["W"] = dfPfoData["shower"]["general"].query(preFilters["W"])
dfPfoData["shower"]["3D"] = dfPfoData["shower"]["general"].query(preFilters["3D"])
dfPfoData["track"] = {"general": dfPfoData["general"].query("isShower==0")}
dfPfoData["track"]["U"] = dfPfoData["track"]["general"].query(preFilters["U"])
dfPfoData["track"]["V"] = dfPfoData["track"]["general"].query(preFilters["V"])
dfPfoData["track"]["W"] = dfPfoData["track"]["general"].query(preFilters["W"])
dfPfoData["track"]["3D"] = dfPfoData["track"]["general"].query(preFilters["3D"])

print((
    "Analysing features using the following samples:\n" +
    "General: %s tracks, %s showers\n" +
    "U View: %s tracks, %s showers\n" +
    "V View: %s tracks, %s showers\n" +
    "W View: %s tracks, %s showers\n" +
    "3D View: %s tracks, %s showers\n") %
    (
        len(dfPfoData["track"]["general"]), len(dfPfoData["shower"]["general"]),
        len(dfPfoData["track"]["U"]), len(dfPfoData["shower"]["U"]),
        len(dfPfoData["track"]["V"]), len(dfPfoData["shower"]["V"]),
        len(dfPfoData["track"]["W"]), len(dfPfoData["shower"]["W"]),
        len(dfPfoData["track"]["3D"]), len(dfPfoData["shower"]["3D"]),
    )
)

for feature in features:
    if efficiencyPurityPlots["plot"]:
        dfTrackData = dfPfoData["track"][GetFeatureView(feature["name"])]
        dfShowerData = dfPfoData["shower"][GetFeatureView(feature["name"])]
        # Get optimal purity and efficiency
        testValues = np.linspace(feature["bins"][0], feature["bins"][-1], efficiencyPurityPlots["nTestValues"])
        (
            bestTrackCutoff, trackEfficiencies, trackPurities, trackPurityEfficiencies, 
            bestShowerCutoff, showerEfficiencies, showerPurities, showerPurityEfficiencies
        ) = OptimiseCutoff(dfTrackData, dfShowerData, feature['name'], testValues, feature['showerCutDirection'])

        # Printing results for optimal purity and efficiency
        print("Performance results for %s:" % feature['name'])
        print("\nOptimal track cutoff %.3f" % bestTrackCutoff)
        PrintPurityEfficiency(dfTrackData, dfShowerData, feature['name'], bestTrackCutoff, feature['showerCutDirection'])
        print("\nOptimal shower cutoff %.3f" % bestShowerCutoff)
        PrintPurityEfficiency(dfTrackData, dfShowerData, feature['name'], bestShowerCutoff, feature['showerCutDirection'])
    if featureHistograms["plot"]:
        # Plot histograms
        feature['filters'] = featureHistograms["filters"]
        feature['yAxis'] = 'log'
        fig, ax = CreateHistogramWire(pd.concat((dfTrackData, dfShowerData)), feature)
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

featureNames = [feature['name'] for feature in features]
dfPfoDataCorr = dfPfoData["general"]
for feature in features:
    dfPfoDataCorr = dfPfoDataCorr.query(preFilters[GetFeatureView(feature["name"])])
rMatrix = dfPfoDataCorr[featureNames].corr()
rSquaredMatrix = rMatrix * rMatrix
sn.heatmap(rSquaredMatrix, annot=True, annot_kws={"size": 20}, cmap="Blues")
plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
plt.tight_layout()
plt.savefig("FeatureRSquaredMatrix.svg", format='svg', dpi=1200)
plt.show()