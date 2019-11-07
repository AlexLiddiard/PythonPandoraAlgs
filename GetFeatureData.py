import os
import UpRootFileReader
import TrackShowerFeatures.TrackShowerFeature0 as tsf0
import TrackShowerFeatures.TrackShowerFeature1 as tsf1
import TrackShowerFeatures.TrackShowerFeature2 as tsf2
import pandas as pd

minHits = 2

if __name__ == "__main__":
    #directory = input("Enter a folder path containing ROOT files: ")
    #fileList = os.listdir(directory)
    fileList = os.listdir('/home/jack/Documents/Pandora/PythonPandoraAlgs/ROOT Files/')

    print("EventId\tPfoId\tType\t[Features]")
    
    pfoFeatureList = []
    
    for fileName in fileList:
        #events = UpRootFileReader.ReadRootFile(os.path.join(directory, fileName))
        events = UpRootFileReader.ReadRootFile(os.path.join('/home/jack/Documents/Pandora/PythonPandoraAlgs/ROOT Files/', fileName))

        
        for eventPfos in events:
            for pfo in eventPfos:
                pfoTrueType = pfo.TrueTypeW()

                if pfoTrueType == -1 or pfo.nHitsW < minHits:
                    continue
                
                featureDictionary = {}
                
                featureDictionary['fileName'] = fileName
                featureDictionary['eventId'] = pfo.eventId
                featureDictionary['pfoId'] = pfo.pfoId
                featureDictionary['pfoTrueType'] = pfoTrueType
                featureDictionary['F0a'] = tsf0.GetFeature(pfo)
                featureDictionary['F1a'] = tsf1.GetFeature(pfo)
                F2a, F2b = tsf2.GetFeature(pfo)
                featureDictionary['F2a'] = F2a
                featureDictionary['F2b'] = F2b
                
                pfoFeatureList.append(featureDictionary)
                
    df = pd.DataFrame(pfoFeatureList)
    df.to_pickle('featureData.pickle')
    