import pandas as pd
from tqdm import tqdm
from glob import glob
import UpRootFileReader
import TrackShowerFeatures.TrackShowerFeature0 as tsf0
import TrackShowerFeatures.TrackShowerFeature1 as tsf1
import TrackShowerFeatures.TrackShowerFeature2 as tsf2
import TrackShowerFeatures.TrackShowerFeature3 as tsf3

myTestArea = "/home/tomalex/Pandora"
rootFileDirectory = myTestArea + "/PythonPandoraAlgs/ROOT Files"
outputPickleFile = myTestArea + '/PythonPandoraAlgs/featureDataTemp.bz2'
wireViews = (True, True, True)

if __name__ == "__main__":
    filePaths =  glob(rootFileDirectory + '/**/*.root', recursive=True)
    pfoData = []
    for filePath in tqdm(filePaths):
        events = UpRootFileReader.ReadRootFile(filePath)

        for eventPfos in events:
            for pfo in eventPfos:
                if abs(pfo.mcPdgCodeW) in (0, 14, 12) or pfo.nHitsPfoThreeD == 0:
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
                    'maxCoordX': min(pfo.xCoordThreeD),
                    'maxCoordY': min(pfo.yCoordThreeD),
                    'maxCoordZ': min(pfo.zCoordThreeD)
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
                pfoDataDict.update(tsf0.GetFeature(pfo, wireViews))
                pfoDataDict.update(tsf1.GetFeature(pfo, wireViews))
                pfoDataDict.update(tsf2.GetFeature(pfo, wireViews))
                pfoDataDict.update(tsf3.GetFeature(pfo, wireViews))
                pfoData.append(pfoDataDict)

    df = pd.DataFrame(pfoData)
    df.to_pickle('featureData.pickle')

    print('\n\n Finished!')
