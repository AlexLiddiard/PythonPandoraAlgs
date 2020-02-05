import BaseConfig as bc
import pandas as pd
import concurrent.futures as cf
from tqdm import tqdm
from glob import glob
import UpRootFileReader
import importlib
import numpy as np
from itertools import count
import GetFeatureDataConfig as cfg
algorithms = [importlib.import_module(algorithm) for algorithm in cfg.algorithmNames]

def ProcessFile(filePath):
    events = UpRootFileReader.ReadRootFile(filePath)
    algorithmData = [[] for i in range(len(algorithms))]
    for eventPfos in events:
        for pfo in eventPfos:
            if abs(pfo.mcPdgCode) in (0, 14, 12) or pfo.nHitsPfo3D == 0:
                continue
            for data, algorithm in zip(algorithmData, algorithms):
                data.append(algorithm.GetFeatures(pfo, cfg.calculateViews))
    return [pd.DataFrame(data) for data in algorithmData]

if __name__ == "__main__":
    filePaths =  glob(cfg.rootFileDirectory + '/**/*.root', recursive=True)
    if filePaths:
        with cf.ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(ProcessFile, filePaths), total=len(filePaths)))
        for i, algorithmName in zip(count(0), cfg.algorithmNames):
            df = pd.concat([result[i] for result in results]).reset_index(drop=True)
            df.to_pickle(bc.dataFolderFull + "/" + cfg.outputDataName + "_" + algorithmName + ".pickle")
        print('\nFinished!')
    else:
        print('No ROOT files found!')


