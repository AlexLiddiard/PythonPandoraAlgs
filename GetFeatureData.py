import pandas as pd
from tqdm import tqdm
from glob import glob
import UpRootFileReader
import TrackShowerFeatures.TrackShowerFeature0 as tsf0
import TrackShowerFeatures.TrackShowerFeature1 as tsf1
import TrackShowerFeatures.TrackShowerFeature2 as tsf2
import TrackShowerFeatures.TrackShowerFeature3 as tsf3

myTestArea = "/home/alexliddiard/Desktop/Pandora"
rootFileDirectory = myTestArea + "/PythonPandoraAlgs/ROOT Files"
outputPickleFile = myTestArea + '/PythonPandoraAlgs/featureDataTemp.pickle'

if __name__ == "__main__":
    filePaths =  glob(rootFileDirectory + '/**/*.root', recursive=True)
    pfoFeatureList = []
    uMin = 1000
    uMax = 0
    vMin = 1000
    vMax = 0
    wMin = 1000
    wMax = 0
    xMin = 1000
    xMax = 0
    for filePath in tqdm(filePaths):
        events = UpRootFileReader.ReadRootFile(filePath)

        for eventPfos in events:
            for pfo in eventPfos:
                if len(pfo.wireCoordU) > 0:
                    uMin = min(uMin, min(pfo.wireCoordU))
                    uMax = max(uMax, max(pfo.wireCoordU))
                if len(pfo.wireCoordV) > 0:
                    vMin = min(vMin, min(pfo.wireCoordV))
                    vMax = max(vMax, max(pfo.wireCoordV))
                if len(pfo.wireCoordW) > 0:
                    wMin = min(wMin, min(pfo.wireCoordW))
                    wMax = max(wMax, max(pfo.wireCoordW))
                if len(pfo.driftCoordU) > 0:
                    xMin = min(xMin, min(pfo.driftCoordU))
                    xMax = max(xMax, max(pfo.driftCoordU))
                if len(pfo.driftCoordV) > 0:
                    xMin = min(xMin, min(pfo.driftCoordV))
                    xMax = max(xMax, max(pfo.driftCoordV))
                if len(pfo.driftCoordW) > 0:
                    xMin = min(xMin, min(pfo.driftCoordW))
                    xMax = max(xMax, max(pfo.driftCoordW))
                '''
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
                featureDictionary.update(tsf3.GetFeature(pfo))
                pfoFeatureList.append(featureDictionary)
                '''
    print("\nuMin %f, uMax %f" % (uMin, uMax))
    print("vMin %f, vMax %f" % (vMin, vMax))
    print("wMin %f, wMax %f" % (wMin, wMax))
    print("xMin %f, xMax %f" % (xMin, xMax))
'''
    df = pd.DataFrame(pfoFeatureList)
    df.to_pickle('featureData.pickle')

    print('\n\n Finished!')
'''
