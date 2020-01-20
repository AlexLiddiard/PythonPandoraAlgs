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

myTestArea = "/home/tomalex/Pandora/"
rootFileDirectory = myTestArea + "/PythonPandoraAlgs/ROOT Files"
outputPickleFile = myTestArea + '/PythonPandoraAlgs/featureDataTemp.bz2'
calculateViews = {
    "U": True,
    "V": True,
    "W": True,
    "3D": True
}

def ProcessFile(filePath):
    events = UpRootFileReader.ReadRootFile(filePath)
    pfoData = []
    for eventPfos in events:
        for pfo in eventPfos:
            if abs(pfo.mcPdgCode) in (0, 14, 12) or pfo.nHitsPfo3D == 0:
                continue
            pfoDataDict = {
                'fileName': pfo.fileName,
                'eventId': pfo.eventId,
                'pfoId': pfo.pfoId,
                'mcNuanceCode': pfo.mcNuanceCode,
                'mcPdgCode': pfo.mcPdgCode,
                'mcpMomentum': pfo.mcpMomentum,
                'isShower': pfo.IsShower(),
                'minCoordX': min(pfo.xCoord3D),
                'minCoordY': min(pfo.yCoord3D),
                'minCoordZ': min(pfo.zCoord3D),
                'maxCoordX': max(pfo.xCoord3D),
                'maxCoordY': max(pfo.yCoord3D),
                'maxCoordZ': max(pfo.zCoord3D)
            }
            if calculateViews["U"]:
                pfoDataDict.update({
                'nHitsU': pfo.nHitsPfoU,
                'purityU': pfo.PurityU(),
                'completenessU': pfo.CompletenessU()
                })
            if calculateViews["V"]:
                pfoDataDict.update({
                'nHitsV': pfo.nHitsPfoV,
                'purityV': pfo.PurityV(),
                'completenessV': pfo.CompletenessV()
                })
            if calculateViews["W"]:
                pfoDataDict.update({
                'nHitsW': pfo.nHitsPfoW,
                'purityW': pfo.PurityW(),
                'completenessW': pfo.CompletenessW()
                })
            if calculateViews["3D"]:
                pfoDataDict.update({'nHits3D': pfo.nHitsPfo3D})
            
            pfoDataDict.update(lr.GetFeatures(pfo, calculateViews))
            pfoDataDict.update(hb.GetFeatures(pfo, calculateViews))
            pfoDataDict.update(cc.GetFeatures(pfo, calculateViews))
            pfoDataDict.update(asp.GetFeatures(pfo, calculateViews))
            pfoDataDict.update(pca.GetFeatures(pfo, calculateViews))
            pfoDataDict.update(chb.GetFeatures(pfo, calculateViews))
            pfoDataDict.update(csmr.GetFeatures(pfo, calculateViews))
            pfoDataDict.update(bp.GetFeatures(pfo, calculateViews))
            pfoDataDict.update(mr.GetFeatures(pfo, calculateViews))
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


