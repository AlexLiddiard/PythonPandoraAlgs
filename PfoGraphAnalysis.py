import os
import UpRootFileReader as rdr
import scipy as sp
import matplotlib as plt

minHits = 10

if __name__ == "__main__":
    directory = input("Enter a folder path containing ROOT files: ")
    #directory = "/home/jack/Documents/Pandora/PythonPandoraAlgs/ROOT Files/"
    fileList = os.listdir(directory)

    print("EventId\tPfoId\tType\t[Features]")
    for fileName in fileList:
        events = UpRootFileReader.ReadRootFile(os.path.join(directory, fileName))
        events = rdr.ReadRootFile(os.path.join(directory, fileName))    
        for eventPfos in events:
            for pfo in eventPfos:
				
