import svgwrite as sw
import UpRootFileReader as rdr
import glob
import os
import random as rnd
import concurrent.futures as cf
import importlib
from cairosvg import svg2png
from tqdm import tqdm
import numpy as np
from UpRootFileReader import MicroBooneGeo
import CNNConfig as cc
import GeneralConfig as gc
from GetFeatureData import ProcessEvents
from PfoGraphicalAnalyser import FileNameToFilePath
import DataSampler as ds

def EnsureFilePath(filePath):
    if not os.path.exists(filePath):
        os.mkdir(filePath)

def PlotPFOSVG(driftCoords, wireCoords, driftCoordErrors, wireCoordError, energies, fileName, driftSpan = 100, wireSpan = 100):
    
    if len(driftCoords) == 0:
        return

    #driftMinus = driftCoords - driftCoordErrors
    #minDriftBoundary = min(driftMinus)

    #driftPlus = driftCoords + driftCoordErrors
    #maxDriftBoundary = max(driftPlus)

    #minWireCoord = min(wireCoords)
    #maxWireCoord = max(wireCoords)

    halfDriftSpan = driftSpan/2
    halfWireSpan = wireSpan/2
    meanDriftCoord = np.mean(driftCoords)
    meanWireCoord = np.mean(wireCoords)

    #Centre = pga.GetSquareRegionAxesLimits(minDriftBoundary, maxDriftBoundary, minWireCoord - 0.3, maxWireCoord + 0.3)

    #print("%s %s %s %s" %(Centre))
    #dwg = sw.Drawing(filename=fileName, size = (2560, 2560), viewBox = "%s %s %s %s" %(minDriftBoundary, minWireCoord - 0.3, maxDriftBoundary - minDriftBoundary, maxWireCoord + 0.6 - minWireCoord), debug=True)
    dwg = sw.Drawing(filename=fileName, size = cc.imageSizePixels, viewBox = "%s %s %s %s" %(meanDriftCoord - halfDriftSpan, meanWireCoord - halfWireSpan, driftSpan, wireSpan), debug=True)
    dwg.add(dwg.rect(insert=(meanDriftCoord - halfDriftSpan, meanWireCoord - halfWireSpan), size=(driftSpan, wireSpan), rx=None, ry=None, fill='rgb(0,0,0)'))
    for driftCoord, wireCoord, driftCoordError, energy in zip(driftCoords, wireCoords, driftCoordErrors, energies):
        ellipse = dwg.ellipse(center=(driftCoord, wireCoord), r=(driftCoordError, wireCoordError))
        ellipse.fill('white', opacity = min(energy/(wireCoordError * driftCoordError * cc.maxEnergyDensity) , 1)) #min(np.log1p(energy)/cc.logMaxEnergy, 1)
        dwg.add(ellipse)
    svg2png(bytestring=dwg.tostring(), write_to=fileName)

def ProcessFile(fileName):
    events = rdr.ReadRootFile(nameToPathDict[fileName])
    df = dfPfoData.query("fileName == @fileName")
    for className, classQuery in gc.classes.items():
        if className == "all":
            continue
        dfClass = df.query(classQuery)
        for index, pfoGeneralInfo in dfClass.iterrows():
            pfo = events[pfoGeneralInfo.eventId][pfoGeneralInfo.pfoId]
            PlotPFOSVG(pfo.driftCoordW, pfo.wireCoordW, pfo.driftCoordErrW, pfo.wireCoordErr, pfo.energyW, "%s/%s_%s_%s_v001.png" %(currentOutputFolder %(className), pfo.fileName, pfo.eventId, pfo.pfoId), *cc.imageSpan["W"])
            PlotPFOSVG(pfo.driftCoordU, pfo.wireCoordU, pfo.driftCoordErrU, pfo.wireCoordErr, pfo.energyU, "%s/%s_%s_%s_v002.png" %(currentOutputFolder %(className), pfo.fileName, pfo.eventId, pfo.pfoId), *cc.imageSpan["U"])
            PlotPFOSVG(pfo.driftCoordV, pfo.wireCoordV, pfo.driftCoordErrV, pfo.wireCoordErr, pfo.energyV, "%s/%s_%s_%s_v003.png" %(currentOutputFolder %(className), pfo.fileName, pfo.eventId, pfo.pfoId), *cc.imageSpan["V"])

def ProcessSample(dfPfoData, nameToPathDict, sampleName):
    fileNames = list(nameToPathDict.keys() & set(dfPfoData["fileName"]))
    global currentOutputFolder
    currentOutputFolder = cc.outputFolder + "/%s/" + sampleName
    if fileNames:
        with cf.ProcessPoolExecutor() as executor:
            list(tqdm(executor.map(ProcessFile, fileNames), total=len(fileNames)))
    else:
        print('No ROOT files found for ' + sampleName + ' sample!')

EnsureFilePath(cc.outputFolder)

for className in gc.classNames:
    EnsureFilePath(cc.outputFolder + "/" + className)
    EnsureFilePath(cc.outputFolder + "/" + className + "/train")
    EnsureFilePath(cc.outputFolder + "/" + className + "/test")

filePaths = glob.glob(cc.rootFileDirectory + '/**/*.root', recursive=True)
nameToPathDict = FileNameToFilePath(filePaths)
ds.LoadPfoData([]) # Load just the data in GeneralInfo

# Training set
dfPfoData = ds.GetFilteredPfoData("training", "all", "training", "union")
ProcessSample(dfPfoData, nameToPathDict, "train")

# Testing set
dfPfoData = ds.GetFilteredPfoData("performance", "all", "performance", "union")
ProcessSample(dfPfoData, nameToPathDict, "test")
