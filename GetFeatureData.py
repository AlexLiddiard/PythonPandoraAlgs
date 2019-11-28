import os
import UpRootFileReader
import TrackShowerFeatures.TrackShowerFeature0 as tsf0
import TrackShowerFeatures.TrackShowerFeature1 as tsf1
import TrackShowerFeatures.TrackShowerFeature2 as tsf2
import pandas as pd
from tqdm import tqdm

minHits = 2

if __name__ == "__main__":
    directory = input("Enter a folder path containing ROOT files: ")
    #directory = '/home/jack/Documents/Pandora/PythonPandoraAlgs/ROOT Files/'
    fileList = os.listdir(directory)

    pfoFeatureList = []

    for fileName in fileList:
        events = UpRootFileReader.ReadRootFile(os.path.join(directory, fileName))


        for eventPfos in events:
            for pfo in eventPfos:
                pfoTrueType = pfo.TrueTypeW()

                if pfoTrueType == -1 or pfo.nHitsPfoW < minHits:
                    continue

                featureDictionary = {}

                featureDictionary['fileName'] = fileName
                featureDictionary['eventId'] = pfo.eventId
                featureDictionary['pfoId'] = pfo.pfoId
                featureDictionary['pfoTrueType'] = pfoTrueType
                featureDictionary.update(tsf0.GetFeature(pfo))
                featureDictionary.update(tsf1.GetFeature(pfo))
                featureDictionary.update(tsf2.GetFeature(pfo))

                pfoFeatureList.append(featureDictionary)

    df = pd.DataFrame(pfoFeatureList)
    df.to_pickle('featureData.pickle')

    print('\n\n Finished!')
