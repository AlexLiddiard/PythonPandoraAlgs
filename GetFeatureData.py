import pandas as pd
from tqdm import tqdm
from glob import glob
import UpRootFileReader
import TrackShowerFeatures.TrackShowerFeature0 as tsf0
import TrackShowerFeatures.TrackShowerFeature1 as tsf1
import TrackShowerFeatures.TrackShowerFeature2 as tsf2

myTestArea = "/home/alexliddiard/Desktop/Pandora"
rootFileDirectory = myTestArea + "/PythonPandoraAlgs"
outputPickleFile = myTestArea + '/PythonPandoraAlgs/featureDataTemp.pickle'

if __name__ == "__main__":
    filePaths =  glob(rootFileDirectory + '/**/*.root', recursive=True)
    pfoFeatureList = []
    for filePath in tqdm(filePaths):
        events = UpRootFileReader.ReadRootFile(filePath)

        for eventPfos in events:
            for pfo in eventPfos:
                if pfo.monteCarloPDGW == 0 or pfo.nHitsPfoW == 0:
                    continue
                featureDictionary = {
                    'fileName': pfo.fileName,
                    'eventId': pfo.eventId,
                    'pfoId': pfo.pfoId,
                    'absPdgCode': abs(pfo.monteCarloPDGW),
                    'isShower': pfo.IsShowerW(),
                    'nHitsW': pfo.nHitsPfoW,
                    'purityW': pfo.PurityW(),
                    'completenessW': pfo.CompletenessW()
                }
                featureDictionary.update(tsf0.GetFeature(pfo))
                featureDictionary.update(tsf1.GetFeature(pfo))
                featureDictionary.update(tsf2.GetFeature(pfo))

                pfoFeatureList.append(featureDictionary)

    df = pd.DataFrame(pfoFeatureList)
    df.to_pickle('featureData.pickle')

    print('\n\n Finished!')
