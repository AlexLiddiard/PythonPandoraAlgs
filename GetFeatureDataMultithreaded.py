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

myTestArea = "/home/tomalex/Pandora"
rootFileDirectory = myTestArea + "/PythonPandoraAlgs/ROOT Files"
outputPickleFile = myTestArea + '/PythonPandoraAlgs/featureData.bz2'
wireViews = (True, True, True)

def ProcessFile(filePath):
    events = UpRootFileReader.ReadRootFile(filePath)
    pfoData = []
    for eventPfos in events:
        for pfo in eventPfos:
            if abs(pfo.mcPdgCode) in (0, 14, 12) or pfo.nHitsPfoThreeD == 0 :
                continue
            pfoDataDict = {
                'fileName': pfo.fileName,
                'eventId': pfo.eventId,
                'pfoId': pfo.pfoId,
                'absPdgCode': abs(pfo.mcPdgCode),
                'isShower': pfo.IsShower(),
                'minCoordX': min(pfo.xCoordThreeD),
                'minCoordY': min(pfo.yCoordThreeD),
                'minCoordZ': min(pfo.zCoordThreeD),
                'maxCoordX': max(pfo.xCoordThreeD),
                'maxCoordY': max(pfo.yCoordThreeD),
                'maxCoordZ': max(pfo.zCoordThreeD)
            }
            if wireViews[0]:
                pfoDataDict.update({
                'nHitsU': pfo.nHitsPfoU,
                'purityU': pfo.PurityU(),
                'completenessU': pfo.CompletenessU()
                })
            if wireViews[1]:
                pfoDataDict.update({
                'nHitsV': pfo.nHitsPfoV,
                'purityV': pfo.PurityV(),
                'completenessV': pfo.CompletenessV()
                })
            if wireViews[2]:
                pfoDataDict.update({
                'nHitsW': pfo.nHitsPfoW,
                'purityW': pfo.PurityW(),
                'completenessW': pfo.CompletenessW()
                })
            pfoDataDict.update(lr.GetFeatures(pfo, wireViews))
            pfoDataDict.update(hb.GetFeatures(pfo, wireViews))
            pfoDataDict.update(cc.GetFeatures(pfo, wireViews))
            pfoDataDict.update(asp.GetFeatures(pfo, wireViews))
            pfoDataDict.update(pca.GetFeatures(pfo, wireViews))
            pfoDataDict.update(chb.GetFeatures(pfo, wireViews))
            pfoDataDict.update(csmr.GetFeatures(pfo, wireViews))
            pfoData.append(pfoDataDict)
    return pd.DataFrame(pfoData)

if __name__ == "__main__":
    filePaths =  glob(rootFileDirectory + '/**/*.root', recursive=True)
    if filePaths:
        with cf.ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(ProcessFile, filePaths), total=len(filePaths)))
        df = pd.concat(results).sample(frac=1).reset_index(drop=True) # Shuffle the data to avoid ordering bias during our analysis
        df.to_pickle(outputPickleFile)
        print('\nFinished!')
    else:
        print('No ROOT files found!')


