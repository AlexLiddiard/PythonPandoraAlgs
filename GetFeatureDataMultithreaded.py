import pandas as pd
import concurrent.futures as cf
from tqdm import tqdm
from glob import glob
import UpRootFileReader
import TrackShowerFeatures.LinearRegression as lr
import TrackShowerFeatures.HitBinning as hb
import TrackShowerFeatures.ChainCreation as cc
import TrackShowerFeatures.AngularSpan as asp
import TrackShowerFeatures.PCAnalysis as pca
import TrackShowerFeatures.ChargedHitBinning as chb
import TrackShowerFeatures.ChargeStdMeanRatio as csmr
import TrackShowerFeatures.BraggPeak as bp
import TrackShowerFeatures.MoliereRadius as mr
import importlib
import numpy as np
from itertools import count

myTestArea = "/home/epp/phuznm/Documents/Pandora/"
rootFileDirectory = myTestArea + "/PandoraCoW/BNBNuOnly"
outputDataFolder = myTestArea + '/PythonPandoraAlgs/TrackShowerData/'
outputDataName = "BNBNuOnly"
algorithmFolder = "TrackShowerFeatures"
algorithmNames = (
    "GeneralInfo",
    "LinearRegression",
    "HitBinning",
    "ChainCreation",
    "AngularSpan",
    "PCAnalysis",
    "ChargedHitBinning",
    "ChargeStdMeanRatio",
    "BraggPeak",
    "MoliereRadius"
)
calculateViews = {
    "U": True,
    "V": True,
    "W": True,
    "3D": True
}
algorithms = [importlib.import_module(algorithmFolder + "." + algorithm) for algorithm in algorithmNames]

def ProcessFile(filePath):
    events = UpRootFileReader.ReadRootFile(filePath)
    algorithmData = [[] for i in range(len(algorithms))]
    for eventPfos in events:
        for pfo in eventPfos:
            if abs(pfo.mcPdgCode) in (0, 14, 12) or pfo.nHitsPfo3D == 0:
                continue
            for data, algorithm in zip(algorithmData, algorithms):
                data.append(algorithm.GetFeatures(pfo, calculateViews))
    return [pd.DataFrame(data) for data in algorithmData]

if __name__ == "__main__":
    filePaths =  glob(rootFileDirectory + '/**/*.root', recursive=True)
    if filePaths:
        with cf.ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(ProcessFile, filePaths), total=len(filePaths)))

        idx = None
        for i, algorithmName in zip(count(0), algorithmNames):
            df = pd.concat([result[i] for result in results]).reset_index(drop=True)
            df.to_pickle(outputDataFolder + outputDataName + "_" + algorithmName + ".pickle")
        print('\nFinished!')
    else:
        print('No ROOT files found!')


