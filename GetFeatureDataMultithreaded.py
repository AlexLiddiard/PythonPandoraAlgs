import os
import glob
import UpRootFileReader
import TrackShowerFeatures.TrackShowerFeature0 as tsf0
import TrackShowerFeatures.TrackShowerFeature1 as tsf1
import TrackShowerFeatures.TrackShowerFeature2 as tsf2
import pandas as pd
import concurrent.futures
from tqdm import tqdm

minHits = 2
dataFileName = 'featureData.pickle'

def ProcessFile(filePath):
    events = UpRootFileReader.ReadRootFile(filePath)
    pfoFeatureList = []
    for eventPfos in events:
        for pfo in eventPfos:
            pfoTrueType = pfo.TrueTypeW()
            if pfoTrueType == -1 or pfo.nHitsW < minHits:
                continue
            featureDictionary = {
                'fileName': os.path.basename(filePath),
                'eventId': pfo.eventId,
                'pfoId': pfo.pfoId,
                'pfoTrueType': pfoTrueType
            }
            featureDictionary.update(tsf0.GetFeature(pfo))
            featureDictionary.update(tsf1.GetFeature(pfo))
            featureDictionary.update(tsf2.GetFeature(pfo))
            pfoFeatureList.append(featureDictionary)
    return pd.DataFrame(pfoFeatureList)

if __name__ == "__main__":
    directory = input("Enter a folder path containing ROOT files: ")
    filePaths =  glob.glob(directory + '/**/*.root', recursive=True)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(ProcessFile, filePaths), total=len(filePaths)))
    df = pd.concat(results)
    df.to_pickle(dataFileName)
    print('\nFinished!')
