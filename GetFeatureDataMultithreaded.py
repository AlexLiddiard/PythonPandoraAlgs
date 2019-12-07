import pandas as pd
import concurrent.futures as cf
from tqdm import tqdm
from glob import glob
import UpRootFileReader
import TrackShowerFeatures.TrackShowerFeature0 as tsf0
import TrackShowerFeatures.TrackShowerFeature1 as tsf1
import TrackShowerFeatures.TrackShowerFeature2 as tsf2
import TrackShowerFeatures.TrackShowerFeature3 as tsf3

myTestArea = "/home/epp/phuznm/Documents/Pandora/"
rootFileDirectory = myTestArea + "PandoraCoW/"
outputPickleFile = myTestArea + '/PythonPandoraAlgs/featureData.bz2'
wireViews = (True, True, True)

def ProcessFile(filePath):
    events = UpRootFileReader.ReadRootFile(filePath)
    pfoData = []
    for eventPfos in events:
        for pfo in eventPfos:
            if abs(pfo.mcPdgCodeW) in (0, 14, 12) or pfo.nHitsPfoThreeD == 0 :
                continue
            pfoDataDict = {
                'fileName': pfo.fileName,
                'eventId': pfo.eventId,
                'pfoId': pfo.pfoId,
                'absPdgCode': abs(pfo.mcPdgCodeW),
                'isShower': pfo.IsShowerW(),
                'minCoordX': min(pfo.xCoordThreeD),
                'minCoordY': min(pfo.yCoordThreeD),
                'minCoordZ': min(pfo.zCoordThreeD),
                'maxCoordX': max(pfo.xCoordThreeD),
                'maxCoordY': max(pfo.yCoordThreeD),
                'maxCoordZ': max(pfo.zCoordThreeD)
            }
            #print(min(pfo.yCoordThreeD))
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
            pfoDataDict.update(tsf0.GetFeature(pfo, wireViews))
            pfoDataDict.update(tsf1.GetFeature(pfo, wireViews))
            pfoDataDict.update(tsf2.GetFeature(pfo, wireViews))
            pfoDataDict.update(tsf3.GetFeature(pfo, wireViews))
            pfoData.append(pfoDataDict)
    return pd.DataFrame( pfoData)

if __name__ == "__main__":
    filePaths =  glob(rootFileDirectory + '/**/*.root', recursive=True)
    if filePaths:
        with cf.ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(ProcessFile, filePaths), total=len(filePaths)))
        df = pd.concat(results)
        df.to_pickle(outputPickleFile)
        print('\nFinished!')
    else:
        print('No ROOT files found!')


