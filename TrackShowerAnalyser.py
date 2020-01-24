import FeatureAnalyser as fa
import DataSampler as ds
import numpy as np

features = (
    #{'name': 'RSquaredU', 'algorithmName': 'LinearRegression', 'bins': np.linspace(0, 1, num=50), 'showerCutDirection': 'left'},
    #{'name': 'RSquaredV', 'algorithmName': 'LinearRegression', 'bins': np.linspace(0, 1, num=50), 'showerCutDirection': 'left'},
    #{'name': 'RSquaredW', 'algorithmName': 'LinearRegression', 'bins': np.linspace(0, 1, num=50), 'showerCutDirection': 'left'},
    #{'name': 'BinnedHitStdU', 'algorithmName': 'HitBinning', 'bins': np.linspace(0, 12, num=50), 'showerCutDirection': 'right'},
    #{'name': 'BinnedHitStdV', 'algorithmName': 'HitBinning', 'bins': np.linspace(0, 12, num=50), 'showerCutDirection': 'right'},
    #{'name': 'BinnedHitStdW', 'algorithmName': 'HitBinning', 'bins': np.linspace(0, 12, num=50), 'showerCutDirection': 'right'},
    #{'name': 'RadialBinStdU', 'algorithmName': 'HitBinning', 'bins': np.linspace(0, 12, num=50), 'showerCutDirection': 'right'},
    #{'name': 'RadialBinStdV', 'algorithmName': 'HitBinning', 'bins': np.linspace(0, 12, num=50), 'showerCutDirection': 'right'},
    #{'name': 'RadialBinStdW', 'algorithmName': 'HitBinning', 'bins': np.linspace(0, 12, num=50), 'showerCutDirection': 'right'},
    #{'name': 'RadialBinStd3D', 'algorithmName': 'HitBinning', 'bins': np.linspace(0, 12, num=50), 'showerCutDirection': 'right'},
    #{'name': 'ChainCountU', 'algorithmName': 'ChainCreation', 'bins': np.linspace(1, 50, num=50), 'showerCutDirection': 'right'},
    #{'name': 'ChainCountV', 'algorithmName': 'ChainCreation', 'bins': np.linspace(1, 50, num=50), 'showerCutDirection': 'right'},
    #{'name': 'ChainCountW', 'algorithmName': 'ChainCreation', 'bins': np.linspace(1, 50, num=50), 'showerCutDirection': 'right'},
    #{'name': 'ChainRatioAvgU', 'algorithmName': 'ChainCreation', 'bins': np.linspace(0, 1, num=50), 'showerCutDirection': 'left'},
    #{'name': 'ChainRatioAvgV', 'algorithmName': 'ChainCreation', 'bins': np.linspace(0, 1, num=50), 'showerCutDirection': 'left'},
    #{'name': 'ChainRatioAvgW', 'algorithmName': 'ChainCreation', 'bins': np.linspace(0, 1, num=50), 'showerCutDirection': 'left'},
    #{'name': 'ChainRSquaredAvgU', 'algorithmName': 'ChainCreation', 'bins': np.linspace(0, 1, num=50), 'showerCutDirection': 'left'},
    #{'name': 'ChainRSquaredAvgV', 'algorithmName': 'ChainCreation', 'bins': np.linspace(0, 1, num=50), 'showerCutDirection': 'left'},
    #{'name': 'ChainRSquaredAvgW', 'algorithmName': 'ChainCreation', 'bins': np.linspace(0, 1, num=50), 'showerCutDirection': 'left'},
    #{'name': 'ChainRatioStdU', 'algorithmName': 'ChainCreation', 'bins': np.linspace(0, 0.8, num=50), 'showerCutDirection': 'right'},
    #{'name': 'ChainRatioStdV', 'algorithmName': 'ChainCreation', 'bins': np.linspace(0, 0.8, num=50), 'showerCutDirection': 'right'},
    #{'name': 'ChainRatioStdW', 'algorithmName': 'ChainCreation', 'bins': np.linspace(0, 0.8, num=50), 'showerCutDirection': 'right'},
    #{'name': 'ChainRSquaredStdU', 'algorithmName': 'ChainCreation', 'bins': np.linspace(0, 0.8, num=50), 'showerCutDirection': 'right'},
    #{'name': 'ChainRSquaredStdV', 'algorithmName': 'ChainCreation', 'bins': np.linspace(0, 0.8, num=50), 'showerCutDirection': 'right'},
    #{'name': 'ChainRSquaredStdW', 'algorithmName': 'ChainCreation', 'bins': np.linspace(0, 0.8, num=50), 'showerCutDirection': 'right'},
    #{'name': 'AngularSpanU', 'algorithmName': 'AngularSpan', 'bins': np.linspace(0, m.pi, num=50), 'showerCutDirection': 'right'},
    #{'name': 'AngularSpanV', 'algorithmName': 'AngularSpan', 'bins': np.linspace(0, m.pi, num=50), 'showerCutDirection': 'right'},
    #{'name': 'AngularSpanW', 'algorithmName': 'AngularSpan', 'bins': np.linspace(0, m.pi, num=50), 'showerCutDirection': 'right'},
    #{'name': 'AngularSpan3D', 'algorithmName': 'AngularSpan', 'bins': np.linspace(0, m.pi, num=50), 'showerCutDirection': 'right'},
    #{'name': 'LongitudinalSpanU', 'algorithmName': 'AngularSpan', 'bins': np.linspace(0, 400, num=50), 'showerCutDirection': 'left'},
    #{'name': 'LongitudinalSpanV', 'algorithmName': 'AngularSpan', 'bins': np.linspace(0, 600, num=50), 'showerCutDirection': 'left'},
    #{'name': 'LongitudinalSpanW', 'algorithmName': 'AngularSpan', 'bins': np.linspace(0, 600, num=50), 'showerCutDirection': 'left'},
    #{'name': 'LongitudinalSpan3D', 'algorithmName': 'AngularSpan', 'bins': np.linspace(0, 600, num=50), 'showerCutDirection': 'left'},
    #{'name': 'PcaVarU', 'algorithmName': 'PCAnalysis', 'bins': np.linspace(0, 10, num=50), 'showerCutDirection': 'right'},
    #{'name': 'PcaVarV', 'algorithmName': 'PCAnalysis', 'bins': np.linspace(0, 10, num=50), 'showerCutDirection': 'right'},
    #{'name': 'PcaVarW', 'algorithmName': 'PCAnalysis', 'bins': np.linspace(0, 10, num=50), 'showerCutDirection': 'right'},
    #{'name': 'PcaVar3D', 'algorithmName': 'PCAnalysis', 'bins': np.linspace(0, 10, num=50), 'showerCutDirection': 'right'},
    #{'name': 'PcaRatioU', 'algorithmName': 'PCAnalysis', 'bins': np.linspace(0, 0.4, num=50), 'showerCutDirection': 'right'},
    #{'name': 'PcaRatioV', 'algorithmName': 'PCAnalysis', 'bins': np.linspace(0, 0.4, num=50), 'showerCutDirection': 'right'},
    #{'name': 'PcaRatioW', 'algorithmName': 'PCAnalysis', 'bins': np.linspace(0, 0.4, num=50), 'showerCutDirection': 'right'},
    #{'name': 'PcaRatio3D', 'algorithmName': 'PCAnalysis', 'bins': np.linspace(0, 1, num=50), 'showerCutDirection': 'right'},
    #{'name': 'ChargedBinnedHitStdU', 'algorithmName': 'ChargedHitBinning', 'bins': np.linspace(0, 100, num=25), 'showerCutDirection': 'right'},
    #{'name': 'ChargedBinnedHitStdV', 'algorithmName': 'ChargedHitBinning', 'bins': np.linspace(0, 100, num=25), 'showerCutDirection': 'right'},
    #{'name': 'ChargedBinnedHitStdW', 'algorithmName': 'ChargedHitBinning', 'bins': np.linspace(0, 100, num=25), 'showerCutDirection': 'right'},
    #{'name': 'ChargedStdMeanRatioU', 'algorithmName': 'ChargedHitBinning', 'bins': np.linspace(0, 4, num=100), 'showerCutDirection': 'right'},
    #{'name': 'ChargedStdMeanRatioV', 'algorithmName': 'ChargedHitBinning', 'bins': np.linspace(0, 4, num=100), 'showerCutDirection': 'right'},
    #{'name': 'ChargedStdMeanRatioW', 'algorithmName': 'ChargedHitBinning', 'bins': np.linspace(0, 4, num=100), 'showerCutDirection': 'right'},
    {'name': 'ChargedStdMeanRatio3D', 'algorithmName': 'ChargedHitBinning', 'bins': np.linspace(0, 4, num=100), 'showerCutDirection': 'right'},
    #{'name': 'BraggPeakU', 'algorithmName': 'BraggPeak', 'bins': np.linspace(0, 1, num=100), 'showerCutDirection': 'left'},
    #{'name': 'BraggPeakV', 'algorithmName': 'BraggPeak', 'bins': np.linspace(0, 1, num=100), 'showerCutDirection': 'left'},
    #{'name': 'BraggPeakW', 'algorithmName': 'BraggPeak', 'bins': np.linspace(0, 1, num=100), 'showerCutDirection': 'left'},
    {'name': 'BraggPeak3D', 'algorithmName': 'BraggPeak', 'bins': np.linspace(0, 1, num=100), 'showerCutDirection': 'left'},
    {'name': 'Moliere3D', 'algorithmName': 'MoliereRadius', 'bins': np.linspace(0, 0.2, num=100), 'showerCutDirection': 'right'},
    #{'name': 'BDTU', 'algorithmName': 'DecisionTreeCalculator', 'bins': np.linspace(-10, 15, num = 200), 'showerCutDirection': 'left'},
    #{'name': 'BDTV', 'algorithmName': 'DecisionTreeCalculator', 'bins': np.linspace(-10, 15, num = 200), 'showerCutDirection': 'left'},
    #{'name': 'BDTW', 'algorithmName': 'DecisionTreeCalculator', 'bins': np.linspace(-10, 15, num = 200), 'showerCutDirection': 'left'},
    #{'name': 'BDT3D', 'algorithmName': 'DecisionTreeCalculator', 'bins': np.linspace(-10, 15, num = 200), 'showerCutDirection': 'left'},
    #{'name': 'BDTMulti', 'algorithmName': 'DecisionTreeCalculator', 'bins': np.linspace(-10, 15, num = 200), 'showerCutDirection': 'left'},
    {'name': 'mcpMomentum', 'algorithmName': 'GeneralInfo', 'bins': np.linspace(0, 0.3, num = 100), 'cutDirection': 'left'},
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
ds.dfPerfPfoData = ds.GetFilteredPfoData(ds.performancePreFilters, features, portion=(ds.trainingFraction, 1))

print("Analysing features using the following samples:\n")
ds.PrintSampleInput(ds.dfPerfPfoData)

for feature in features:
    dfPfoData = ds.dfPerfPfoData["all"][ds.GetFeatureView(feature["name"])]
    dfTrackData = ds.dfPerfPfoData["track"][ds.GetFeatureView(feature["name"])]
    dfShowerData = ds.dfPerfPfoData["shower"][ds.GetFeatureView(feature["name"])]
    cutoffValues, cutoffResults = fa.GetBestPurityEfficiency(dfTrackData, dfShowerData, ("track", "shower"), feature, purityEfficiency["nTestCuts"])
    if featureHistogram["plot"]:
        fa.PlotVariableHistogram(dfPfoData, ("track", "shower"), feature, featureHistogram, cutoffResults[4])
    if purityEfficiency["plot"]:        
        fa.PlotPurityEfficiencyVsCutoff(feature["name"], ("track", "shower"), cutoffValues, cutoffResults)

fa.CorrelationMatrix([feature['name'] for feature in features], ds.GetViewsUsed(features), ds.performancePreFilters, ds.dfPerfPfoData)