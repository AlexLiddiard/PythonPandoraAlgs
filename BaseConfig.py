import matplotlib as mpl
import sys

############################################## BASE CONFIGURATION ##################################################

myTestArea = "/home/tomalex/Pandora/"
pythonFolder = "PythonPandoraAlgs"
analysisFolder = "TrackShower"
pyplotBackend = 'Qt5Agg'

pythonFolderFull = myTestArea + "/" + pythonFolder
analysisFolderFull = myTestArea + "/" + "/" + analysisFolder
featureAlgsFolderFull = myTestArea + "/" + pythonFolder + "/" + analysisFolder + "/FeatureAlgs"
configFolderFull = myTestArea + "/" + pythonFolder + "/" + analysisFolder + "/Config"
dataFolderFull = myTestArea + "/" + pythonFolder + "/" + analysisFolder + "/PickleData"
figureFolderFull = myTestArea + "/" + pythonFolder + "/" + analysisFolder + "/Figures"

############################################## CONFIGURATION PROCESSING ##################################################

sys.path.append(myTestArea)
sys.path.append(pythonFolderFull)
sys.path.append(analysisFolderFull)
sys.path.append(featureAlgsFolderFull)
sys.path.append(configFolderFull)
mpl.use(pyplotBackend)