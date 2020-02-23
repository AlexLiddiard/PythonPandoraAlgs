import BaseConfig as bc
import numpy as np
import DataSamplerConfig as dsc
import GeneralConfig as gc

############################### Image Generator Configuration #################################
rootFileDirectory =  bc.pythonFolderFull + "/ROOT Files"
imageOutputFolder = bc.analysisFolderFull + "/SVGData"
# {"View": (driftSpan, wireSpan)}
imageSizePixels = (224, 224)
imageSpan = {
    "U": (100, 100), 
    "V": (100, 100),
    "W": (100, 100),
}

maxEnergyDensity = 1500 / (0.3 * 0.3)

############################### MVCNN Trainer Configuration ##################################

trainingModelName = "testCNN"
trainingBaseModel = "vgg19"
trainingBatchSize = 48
numPFOs = 0 # 0 => all PFOs
learningRate = 5e-5
weightDecay = 0.001
pretraining = False
trainingImagePath = imageOutputFolder
epochsStage1 = 1
epochsStage2 = 1
modelOutputFolder = bc.analysisFolder + "/TrainedModels/"

############################### MVCNN Tester Configuration ##################################

modelInputFolder = bc.analysisFolder + "/TrainedModels/"
testingModelName = "CurrentBest"
testingBaseModel = "vgg19"
testingImagePath = imageOutputFolder
testingBatchSize = 24
