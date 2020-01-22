import FeatureAnalyser as fa
import DataSampler as ds
import numpy as np

features = (
    #{'name': 'RSquaredU', 'bins': np.linspace(0, 1, num=50), 'showerCutDirection': 'left'},
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
    #{'name': 'Moliere3D', 'bins': np.linspace(0, 0.2, num=100), 'showerCutDirection': 'right'},
    #{'name': 'BDTU', 'bins': np.linspace(-10, 15, num = 200), 'showerCutDirection': 'left'},
    #{'name': 'BDTV', 'bins': np.linspace(-10, 15, num = 200), 'showerCutDirection': 'left'},
    #{'name': 'BDTW', 'bins': np.linspace(-10, 15, num = 200), 'showerCutDirection': 'left'},
    #{'name': 'BDT3D', 'bins': np.linspace(-10, 15, num = 200), 'showerCutDirection': 'left'},
    #{'name': 'BDTMulti', 'bins': np.linspace(-10, 15, num = 200), 'showerCutDirection': 'left'},
    {'name': 'mcpMomentum', 'bins': np.linspace(0, 0.3, num = 100), 'cutDirection': 'left'},
)

featureHistogram = {
    "plot": True,
    "filters": (
        ("Showers", "isShower==1", "", True), 
        #("Tracks", "isShower==0", "", True),
        ("Electrons + Positrons", "abs(mcPdgCode)==11", "isShower==1", False),
        ("Photons", "abs(mcPdgCode)==22",  "isShower==1", False),
        #("Protons", "abs(mcPdgCode)==2212", "isShower==0", False),
        #("Muons", "abs(mcPdgCode)==13", "isShower==0", False),
        #("Charged Pions", "abs(mcPdgCode)==211", "isShower==0", False),
    )   
}
purityEfficiency = {
    "plot": True,
    "nTestCuts": 1000
}

# Load the pickle file.
ds.dfPerfPfoData = ds.GetFilteredPfoData(ds.performancePreFilters, portion=(ds.trainingFraction, 1))

print("Analysing features using the following samples:\n")
ds.PrintSampleInput(ds.dfPerfPfoData)

for feature in features:
    dfPfoData = ds.dfPerfPfoData["all"][ds.GetFeatureView(feature["name"])]
    dfTrackData = ds.dfPerfPfoData["track"][ds.GetFeatureView(feature["name"])]
    dfShowerData = ds.dfPerfPfoData["shower"][ds.GetFeatureView(feature["name"])]
    cutoffValues, cutoffResults = fa.GetBestPurityEfficiency(dfTrackData, dfShowerData, ("track", "shower"), feature, purityEfficiency["nTestCuts"])
    if featureHistogram["plot"]:
        fa.PlotFeatureHistogram(dfPfoData, ("track", "shower"), feature, featureHistogram, cutoffResults[4])
    if purityEfficiency["plot"]:        
        fa.PlotPurityEfficiencyVsCutoff(feature["name"], ("track", "shower"), cutoffValues, cutoffResults)

fa.CorrelationMatrix([feature['name'] for feature in features], ds.GetViewsUsed(features), ds.performancePreFilters, ds.dfPerfPfoData)