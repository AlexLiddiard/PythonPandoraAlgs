import glob
import UpRootFileReader
import TrackShowerFeatures.TrackShowerFeature0 as tsf0
import TrackShowerFeatures.TrackShowerFeature1 as tsf1
import TrackShowerFeatures.TrackShowerFeature2 as tsf2
import pandas as pd
import concurrent.futures
from tqdm import tqdm

minHits = 2
minCompleteness = 0.8
minPurity = 0.8
myTestArea = "/home/alexliddiard/Desktop/Pandora"
rootFileDirectory = myTestArea + "/PythonPandoraAlgs/ROOT Files"
outputPickleFile = myTestArea + '/PythonPandoraAlgs/featureData.pickle'

def ProcessFile(filePath):
    events = UpRootFileReader.ReadRootFile(filePath)
    pfoFeatureList = []
    for eventPfos in events:
        for pfo in eventPfos:
            if pfo.nHitsPfoW < minHits or pfo.PurityW() < minPurity or pfo.CompletenessW() < minCompleteness:
                continue
            featureDictionary = {
                'fileName': pfo.fileName,
                'eventId': pfo.eventId,
                'pfoId': pfo.pfoId,
                'absPdgCode': abs((pfo.monteCarloPDGW),
                'isShower': pfo.IsShowerW()
            }
            featureDictionary.update(tsf0.GetFeature(pfo))
            featureDictionary.update(tsf1.GetFeature(pfo))
            featureDictionary.update(tsf2.GetFeature(pfo))
            pfoFeatureList.append(featureDictionary)
    return pd.DataFrame(pfoFeatureList)

if __name__ == "__main__":
    filePaths =  glob.glob(rootFileDirectory + '/**/*.root', recursive=True)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(ProcessFile, filePaths), total=len(filePaths)))
    df = pd.concat(results)
    df.to_pickle(outputPickleFile)
    print('\nFinished!')
