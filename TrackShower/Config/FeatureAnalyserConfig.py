import numpy as np
import math as m
from UpRootFileReader import MicroBooneGeo

############################################## FEATURE ANALYSER CONFIGURATION ##################################################

features = [
    #{'name': 'RSquaredU', 'algorithmName': 'LinearRegression', 'bins': np.linspace(0, 1, num=50), 'cutDirection': 'left'},
    #{'name': 'RSquaredV', 'algorithmName': 'LinearRegression', 'bins': np.linspace(0, 1, num=50), 'cutDirection': 'left'},
    #{'name': 'RSquaredW', 'algorithmName': 'LinearRegression', 'bins': np.linspace(0, 1, num=50), 'cutDirection': 'left'},
    {'name': 'BinnedHitStdU', 'algorithmName': 'HitBinning', 'bins': np.linspace(0, 12, num=50), 'cutDirection': 'right'},
    {'name': 'BinnedHitStdV', 'algorithmName': 'HitBinning', 'bins': np.linspace(0, 12, num=50), 'cutDirection': 'right'},
    {'name': 'BinnedHitStdW', 'algorithmName': 'HitBinning', 'bins': np.linspace(0, 12, num=50), 'cutDirection': 'right'},
    {'name': 'RadialBinHitStdU', 'algorithmName': 'HitBinning', 'bins': np.linspace(0, 12, num=50), 'cutDirection': 'right'},
    {'name': 'RadialBinHitStdV', 'algorithmName': 'HitBinning', 'bins': np.linspace(0, 12, num=50), 'cutDirection': 'right'},
    {'name': 'RadialBinHitStdW', 'algorithmName': 'HitBinning', 'bins': np.linspace(0, 12, num=50), 'cutDirection': 'right'},
    {'name': 'RadialBinHitStd3D', 'algorithmName': 'HitBinning', 'bins': np.linspace(0, 12, num=50), 'cutDirection': 'right', 'cutPlot': "simple"},
    {'name': 'BinnedChargeStdU', 'algorithmName': 'HitBinning', 'bins': np.linspace(0, 12, num=50), 'cutDirection': 'right'},
    {'name': 'BinnedChargeStdV', 'algorithmName': 'HitBinning', 'bins': np.linspace(0, 12, num=50), 'cutDirection': 'right'},
    {'name': 'BinnedChargeStdW', 'algorithmName': 'HitBinning', 'bins': np.linspace(0, 12, num=50), 'cutDirection': 'right'},
    {'name': 'RadialBinChargeStdU', 'algorithmName': 'HitBinning', 'bins': np.linspace(0, 12, num=50), 'cutDirection': 'right'},
    {'name': 'RadialBinChargeStdV', 'algorithmName': 'HitBinning', 'bins': np.linspace(0, 12, num=50), 'cutDirection': 'right'},
    {'name': 'RadialBinChargeStdW', 'algorithmName': 'HitBinning', 'bins': np.linspace(0, 12, num=50), 'cutDirection': 'right'},
    {'name': 'RadialBinChargeStd3D', 'algorithmName': 'HitBinning', 'bins': np.linspace(0, 12, num=50), 'cutDirection': 'right', 'cutPlot': "simple"},
    #{'name': 'ChainCountU', 'algorithmName': 'ChainCreation', 'bins': np.linspace(1, 50, num=50), 'cutDirection': 'right'},
    #{'name': 'ChainCountV', 'algorithmName': 'ChainCreation', 'bins': np.linspace(1, 50, num=50), 'cutDirection': 'right'},
    #{'name': 'ChainCountW', 'algorithmName': 'ChainCreation', 'bins': np.linspace(1, 50, num=50), 'cutDirection': 'right'},
    #{'name': 'ChainRatioAvgU', 'algorithmName': 'ChainCreation', 'bins': np.linspace(0, 1, num=50), 'cutDirection': 'left'},
    #{'name': 'ChainRatioAvgV', 'algorithmName': 'ChainCreation', 'bins': np.linspace(0, 1, num=50), 'cutDirection': 'left'},
    ##{'name': 'ChainRatioAvgW', 'algorithmName': 'ChainCreation', 'bins': np.linspace(0, 1, num=50), 'cutDirection': 'left', 'cutPlot': "simple"},
    #{'name': 'ChainRSquaredAvgU', 'algorithmName': 'ChainCreation', 'bins': np.linspace(0, 1, num=50), 'cutDirection': 'left'},
    #{'name': 'ChainRSquaredAvgV', 'algorithmName': 'ChainCreation', 'bins': np.linspace(0, 1, num=50), 'cutDirection': 'left'},
    ##{'name': 'ChainRSquaredAvgW', 'algorithmName': 'ChainCreation', 'bins': np.linspace(0, 1, num=50), 'cutDirection': 'left', 'cutPlot': "simple"},
    #{'name': 'ChainRatioStdU', 'algorithmName': 'ChainCreation', 'bins': np.linspace(0, 0.8, num=50), 'cutDirection': 'right'},
    #{'name': 'ChainRatioStdV', 'algorithmName': 'ChainCreation', 'bins': np.linspace(0, 0.8, num=50), 'cutDirection': 'right'},
    #{'name': 'ChainRatioStdW', 'algorithmName': 'ChainCreation', 'bins': np.linspace(0, 0.8, num=50), 'cutDirection': 'right'},
    #{'name': 'ChainRSquaredStdU', 'algorithmName': 'ChainCreation', 'bins': np.linspace(0, 0.8, num=50), 'cutDirection': 'right'},
    #{'name': 'ChainRSquaredStdV', 'algorithmName': 'ChainCreation', 'bins': np.linspace(0, 0.8, num=50), 'cutDirection': 'right'},
    #{'name': 'ChainRSquaredStdW', 'algorithmName': 'ChainCreation', 'bins': np.linspace(0, 0.8, num=50), 'cutDirection': 'right'},
    #{'name': 'AngularSpanU', 'algorithmName': 'AngularSpan', 'bins': np.linspace(0, m.pi, num=50), 'cutDirection': 'right'},
    #{'name': 'AngularSpanV', 'algorithmName': 'AngularSpan', 'bins': np.linspace(0, m.pi, num=50), 'cutDirection': 'right'},
    ##{'name': 'AngularSpanW', 'algorithmName': 'AngularSpan', 'bins': np.linspace(0, m.pi, num=50), 'cutDirection': 'right'},
    #{'name': 'AngularSpan3D', 'algorithmName': 'AngularSpan', 'bins': np.linspace(0, m.pi, num=50), 'cutDirection': 'right', 'cutPlot': "simple"},
    #{'name': 'LongitudinalSpanU', 'algorithmName': 'AngularSpan', 'bins': np.linspace(0, 400, num=50), 'cutDirection': 'left'},
    #{'name': 'LongitudinalSpanV', 'algorithmName': 'AngularSpan', 'bins': np.linspace(0, 600, num=50), 'cutDirection': 'left'},
    ##{'name': 'LongitudinalSpanW', 'algorithmName': 'AngularSpan', 'bins': np.linspace(0, 600, num=50), 'cutDirection': 'left'},
    #{'name': 'LongitudinalSpan3D', 'algorithmName': 'AngularSpan', 'bins': np.linspace(0, 600, num=50), 'cutDirection': 'left'},
    #{'name': 'PcaVarU', 'algorithmName': 'PCAnalysis', 'bins': np.linspace(0, 10, num=50), 'cutDirection': 'right'},
    #{'name': 'PcaVarV', 'algorithmName': 'PCAnalysis', 'bins': np.linspace(0, 10, num=50), 'cutDirection': 'right'},
    #{'name': 'PcaVarW', 'algorithmName': 'PCAnalysis', 'bins': np.linspace(0, 10, num=50), 'cutDirection': 'right'},
    #{'name': 'PcaVar3D', 'algorithmName': 'PCAnalysis', 'bins': np.linspace(0, 10, num=50), 'cutDirection': 'right'},
    #{'name': 'PcaRatioU', 'algorithmName': 'PCAnalysis', 'bins': np.linspace(0, 0.4, num=50), 'cutDirection': 'right'},
    #{'name': 'PcaRatioV', 'algorithmName': 'PCAnalysis', 'bins': np.linspace(0, 0.4, num=50), 'cutDirection': 'right'},
    #{'name': 'PcaRatioW', 'algorithmName': 'PCAnalysis', 'bins': np.linspace(0, 0.4, num=50), 'cutDirection': 'right'},
    #{'name': 'PcaRatio3D', 'algorithmName': 'PCAnalysis', 'bins': np.linspace(0, 0.1, num=50), 'cutDirection': 'right', 'cutPlot': "simple"},
    #{'name': 'ChargedStdMeanRatioU', 'algorithmName': 'ChargeStdMeanRatio', 'bins': np.linspace(0, 4, num=100), 'cutDirection': 'right'},
    #{'name': 'ChargedStdMeanRatioV', 'algorithmName': 'ChargeStdMeanRatio', 'bins': np.linspace(0, 4, num=100), 'cutDirection': 'right'},
    ##{'name': 'ChargedStdMeanRatioW', 'algorithmName': 'ChargeStdMeanRatio', 'bins': np.linspace(0, 4, num=100), 'cutDirection': 'right'},
    #{'name': 'ChargedStdMeanRatio3D', 'algorithmName': 'ChargeStdMeanRatio', 'bins': np.linspace(0, 4, num=100), 'cutDirection': 'right'},
    #{'name': 'BraggPeakU', 'algorithmName': 'BraggPeak', 'bins': np.linspace(0, 1, num=100), 'cutDirection': 'left'},
    #{'name': 'BraggPeakV', 'algorithmName': 'BraggPeak', 'bins': np.linspace(0, 1, num=100), 'cutDirection': 'left'},
    ##{'name': 'BraggPeakW', 'algorithmName': 'BraggPeak', 'bins': np.linspace(0, 1, num=100), 'cutDirection': 'left'},
    #{'name': 'BraggPeak3D', 'algorithmName': 'BraggPeak', 'bins': np.linspace(0, 1, num=50), 'cutDirection': 'left', 'cutPlot': "simple"},
    #{'name': 'Moliere3D', 'algorithmName': 'MoliereRadius', 'bins': np.linspace(0, 0.2, num=50), 'cutDirection': 'right', 'cutPlot': "simple"},
    #{'name': 'BDTU', 'algorithmName': 'BDTCalculator', 'bins': np.linspace(-10, 15, num = 200), 'cutDirection': 'left'},
    #{'name': 'BDTV', 'algorithmName': 'BDTCalculator', 'bins': np.linspace(-10, 15, num = 200), 'cutDirection': 'left'},
    #{'name': 'BDTW', 'algorithmName': 'BDTCalculator', 'bins': np.linspace(-10, 15, num = 200), 'cutDirection': 'left'},
    #{'name': 'BDT3D', 'algorithmName': 'BDTCalculator', 'bins': np.linspace(-10, 15, num = 200), 'cutDirection': 'left'},
    #{'name': 'BDTMulti', 'algorithmName': 'BDTCalculator', 'bins': np.linspace(-10, 15, num = 200), 'cutDirection': 'left'},
    #{'name': 'BDTAll', 'algorithmName': 'BDTCalculator', 'bins': np.linspace(-10, 15, num = 200), 'cutDirection': 'left'},
    #{'name': 'Likelihood', 'algorithmName': 'LikelihoodCalculator', 'bins': np.linspace(0, 1, num = 200), 'cutDirection': 'right'},
    #{'name': 'mcpMomentum', 'algorithmName': 'GeneralInfo', 'bins': np.linspace(0, 2, num = 50), 'cutDirection': 'left', 'plotCutoff': False},
    #{'name': 'nHitsU', 'algorithmName': 'GeneralInfo', 'bins': np.linspace(0, 2000, num = 50), 'cutDirection': 'right', 'cutFixed': 20, 'cutPlot': "simple"},
    #{'name': 'nHitsV', 'algorithmName': 'GeneralInfo', 'bins': np.linspace(0, 2000, num = 50), 'cutDirection': 'right', 'cutFixed': 20, 'cutPlot': "simple"},
    #{'name': 'nHitsW', 'algorithmName': 'GeneralInfo', 'bins': np.linspace(0, 3000, num = 50), 'cutDirection': 'right', 'cutFixed': 20, 'cutPlot': "simple"},
    #{'name': 'nHits3D', 'algorithmName': 'GeneralInfo', 'bins': np.linspace(0, 4000, num = 50), 'cutDirection': 'right', 'cutFixed': 20, 'cutPlot': "simple"},
    #{'name': 'minCoordX3D', 'algorithmName': 'GeneralInfo', 'bins': np.linspace(MicroBooneGeo.RangeX[0], MicroBooneGeo.RangeX[1], num = 50), 'cutDirection': 'left', 'cutFixed': MicroBooneGeo.RangeX[1] - 15, 'cutPlot': "simple"},
    #{'name': 'maxCoordX3D', 'algorithmName': 'GeneralInfo', 'bins': np.linspace(MicroBooneGeo.RangeX[0], MicroBooneGeo.RangeX[1], num = 50), 'cutDirection': 'right', 'cutFixed':  MicroBooneGeo.RangeX[0] + 15, 'cutPlot': "simple"},
    #{'name': 'minCoordY3D', 'algorithmName': 'GeneralInfo', 'bins': np.linspace(MicroBooneGeo.RangeY[0], MicroBooneGeo.RangeY[1], num = 50), 'cutDirection': 'left', 'cutFixed': MicroBooneGeo.RangeY[1] - 20, 'cutPlot': "simple"},
    #{'name': 'maxCoordY3D', 'algorithmName': 'GeneralInfo', 'bins': np.linspace(MicroBooneGeo.RangeY[0], MicroBooneGeo.RangeY[1], num = 50), 'cutDirection': 'right', 'cutFixed': MicroBooneGeo.RangeY[0] + 20, 'cutPlot': "simple"},
    #{'name': 'minCoordZ3D', 'algorithmName': 'GeneralInfo', 'bins': np.linspace(MicroBooneGeo.RangeZ[0], MicroBooneGeo.RangeZ[1], num = 50), 'cutDirection': 'left', 'cutFixed': MicroBooneGeo.RangeZ[1] - 20, 'cutPlot': "simple"},
    #{'name': 'maxCoordZ3D', 'algorithmName': 'GeneralInfo', 'bins': np.linspace(MicroBooneGeo.RangeZ[0], MicroBooneGeo.RangeZ[1], num = 50), 'cutDirection': 'right', 'cutFixed': MicroBooneGeo.RangeZ[0] + 20, 'cutPlot': "simple"},
    #{'name': 'completenessW', 'algorithmName': 'GeneralInfo', 'bins': np.linspace(0, 1, num = 50), 'cutDirection': 'right', 'cutFixed': 0.5, 'cutPlot': "simple"},
    #{'name': 'purityW', 'algorithmName': 'GeneralInfo', 'bins': np.linspace(0, 1, num = 50), 'cutDirection': 'right', 'cutFixed': 0.8, 'cutPlot': "simple"},
]

featureHistogram = {
    "plot": True,
    "filters": [
        ("Showers", "isShower==1", "count", True), 
        ("Tracks", "isShower==0", "count", True),
        #("Correct tracks", "isShower==0 and Likelihood < 0.998", "count", False ),
        #("Incorrect tracks", "isShower==0 and Likelihood > 0.998", "count", False ),
        #("Correct showers", "isShower==1 and Likelihood > 0.998", "count", False ),
        #("Incorrect showers", "isShower==1 and Likelihood < 0.998", "count", False ),
        ("Protons", "mcPdgCode==2212", "count", False ),
        ("Muons", "mcPdgCode==13", "count", False ),
        ("Electrons + Positrons", "abs(mcPdgCode)==11", "count", False),
        ("Photons", "abs(mcPdgCode)==22",  "count", False),
        ("Charged Pions", "abs(mcPdgCode)==211", "count", False),
    ]   
}

purityEfficiencyVsCutoff = {
    "plot": False,
    "nTestCuts": 1000
}