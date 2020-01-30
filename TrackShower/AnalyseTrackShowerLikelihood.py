import DataSampler as ds
import CalculateTrackShowerLikelihood as tsl
import LikelihoodAnalyser as la
import numpy as np

if __name__ == "__main__":
    tsl.features.append({'name': 'Likelihood', 'algorithmName': 'LikelihoodCalculator'})
    ds.GetPerfPfoData(tsl.features)
    print("Analysing track/shower likelihood using the following samples:\n")
    ds.PrintSampleInput(ds.dfPerfPfoData)
    cutoffValues, cutoffResults = la.GetBestPurityEfficiency(
        ds.dfPerfPfoData['track']['union'], 
        ds.dfPerfPfoData['shower']['union'], ('track', 'shower'),
        {'name': 'Likelihood', 'bins': (0, 1), 'cutDirection': 'right'},
        purityEfficiencyVsCutoffGraph['nTestCuts'])

    for histogram in likelihoodHistograms:
        histogram['name'] = 'Likelihood'
        la.PlotVariableHistogram(ds.dfPerfPfoData['all']['union'], ('track', 'shower'), {'name': 'Likelihood', 'cutDirection': 'right'}, histogram, cutoffResults[4])

    la.PlotPurityEfficiencyVsCutoff("Likelihood", ("track", "shower"), cutoffValues, cutoffResults)
    
    for graph in purityEfficiencyBinnedGraphs:
        graph["cutoff"] = cutoffResults[4]
        la.PlotPurityEfficiencyVsVariable(ds.dfPerfPfoData["track"]['union'], ds.dfPerfPfoData["shower"]["union"], ("track", "shower"), graph)