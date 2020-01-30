import sys

myTestArea = "/home/tomalex/Pandora/"
pythonFolder = "PythonPandoraAlgs"
analysisFolder = "ElectronPhoton"


pythonFolderFull = myTestArea + "/" + pythonFolder
analysisFolderFull = myTestArea + "/" + "/" + analysisFolder
featureAlgsFolderFull = myTestArea + "/" + pythonFolder + "/" + analysisFolder + "/FeatureAlgs"
configFolderFull = myTestArea + "/" + pythonFolder + "/" + analysisFolder + "/Config"
dataFolderFull = myTestArea + "/" + pythonFolder + "/" + analysisFolder + "/PickleData"
figureFolderFull = myTestArea + "/" + pythonFolder + "/" + analysisFolder + "/Figures"

sys.path.append(myTestArea)
sys.path.append(pythonFolderFull)
sys.path.append(analysisFolderFull)
sys.path.append(featureAlgsFolderFull)
sys.path.append(configFolderFull)