import os
import UpRootFileReader
import TrackShowerFeatures.TrackShowerFeature0 as tsf0
import TrackShowerFeatures.TrackShowerFeature1 as tsf1
import TrackShowerFeatures.TrackShowerFeature2 as tsf2

minHits = 10

if __name__ == "__main__":
    #directory = input("Enter a folder path containing ROOT files: ")
    #fileList = os.listdir(directory)
    fileList = os.listdir('/home/jack/Documents/Pandora/PythonPandoraAlgs/ROOT Files/')

    print("EventId\tPfoId\tType\t[Features]")
    for fileName in fileList:
        #events = UpRootFileReader.ReadRootFile(os.path.join(directory, fileName))
        events = UpRootFileReader.ReadRootFile(os.path.join('/home/jack/Documents/Pandora/PythonPandoraAlgs/ROOT Files/', fileName))

        
        for eventPfos in events:
            for pfo in eventPfos:
                pfoTrueType = pfo.TrueTypeW()

                if pfoTrueType == -1 or pfo.nHitsW < minHits:
                    continue

                print("%d\t%d\t%d" % (pfo.eventId, pfo.pfoId, pfoTrueType), end="\t")
                print("%.3f" % tsf0.GetFeature(pfo), end="\t")
                print("%.3f" % tsf1.GetFeature(pfo), end="\t")
                print("%.3f" % tsf2.GetFeature(pfo)[1], end="\t")
                print()
