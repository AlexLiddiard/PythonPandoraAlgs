import pandas as pd
import concurrent.futures as cf
from tqdm import tqdm
from glob import glob
import UpRootFileReader
import TrackShowerFeatures.TrackShowerFeature0 as tsf0
import TrackShowerFeatures.TrackShowerFeature1 as tsf1
import TrackShowerFeatures.TrackShowerFeature2 as tsf2
import TrackShowerFeatures.TrackShowerFeature3 as tsf3

myTestArea = "/home/jack/Documents/Pandora"
rootFileDirectory = myTestArea + "/PythonPandoraAlgs/ROOT Files"
outputPickleFile = myTestArea + '/PythonPandoraAlgs/featureDataTemp.pickle'
wireViews = (True, True, True)
def ProcessFile(filePath):
    events = UpRootFileReader.ReadRootFile(filePath)
    pfoFeatureList = []
    for eventPfos in events:
        for pfo in eventPfos:
            if pfo.monteCarloPDGW == 0 or pfo.nHitsPfo() == 0:
                continue
            featureDictionary = {
                'fileName': pfo.fileName,
                'eventId': pfo.eventId,
                'pfoId': pfo.pfoId,
                'absPdgCode': abs(pfo.monteCarloPDGW),
                'isShower': pfo.IsShowerW(),
            }
            if wireViews[0]:
                featureDictionary.update({
                'nHitsU': pfo.nHitsPfoU,
                'purityU': pfo.PurityU(),
                'completenessU': pfo.CompletenessU()
                })
            if wireViews[1]:
                featureDictionary.update({
                'nHitsV': pfo.nHitsPfoV,
                'purityV': pfo.PurityV(),
                'completenessV': pfo.CompletenessV()
                })
            if wireViews[2]:
                featureDictionary.update({
                'nHitsW': pfo.nHitsPfoW,
                'purityW': pfo.PurityW(),
                'completenessW': pfo.CompletenessW()
                })
            featureDictionary.update(tsf0.GetFeature(pfo, wireViews))
            featureDictionary.update(tsf1.GetFeature(pfo, wireViews))
            featureDictionary.update(tsf2.GetFeature(pfo, wireViews))
            featureDictionary.update(tsf3.GetFeature(pfo, wireViews))
            pfoFeatureList.append(featureDictionary)
    return pd.DataFrame(pfoFeatureList)

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
