import BaseConfig as bc
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
from UpRootFileReader import MicroBooneGeo, ProjectVector
import CNNConfig as cc
import GeneralConfig as gc
from GetFeatureData import ProcessEvents
from PfoGraphicalAnalyser import FileNameToFilePath
import DataSampler as ds

def EnsureFilePath(filePath):
    if not os.path.exists(filePath):
        os.mkdir(filePath)

def PlotPFOSVG(driftCoords, wireCoords, driftCoordErrors, wireCoordError, energies, fileName, centre, driftSpan = 100, wireSpan = 100):
    halfDriftSpan = driftSpan/2
    halfWireSpan = wireSpan/2
    dwg = sw.Drawing(filename=fileName, size = cc.imageSizePixels, viewBox = "%s %s %s %s" %(centre[0] - halfDriftSpan, centre[1] - halfWireSpan, driftSpan, wireSpan), debug=True)
    dwg.add(dwg.rect(insert=(centre[0] - halfDriftSpan, centre[1] - halfWireSpan), size=(driftSpan, wireSpan), rx=None, ry=None, fill='rgb(0,0,0)'))
    for driftCoord, wireCoord, driftCoordError, energy in zip(driftCoords, wireCoords, driftCoordErrors, energies):
        ellipse = dwg.ellipse(center=(driftCoord, wireCoord), r=(driftCoordError, wireCoordError))
        ellipse.fill('white', opacity = min(energy/(wireCoordError * driftCoordError * cc.maxEnergyDensity) , 1)) #min(np.log1p(energy)/cc.logMaxEnergy, 1)
        dwg.add(ellipse)
    svg2png(bytestring=dwg.tostring(), write_to=fileName)

def GetImage3DCentre(xCoord3D, yCoord3D, zCoord3D, vertex3D, maxVertexDisplacement):
    centroid3D = np.mean((xCoord3D, yCoord3D, zCoord3D), axis=1)
    if vertex3D is None:
        return centroid3D
    direction = centroid3D - vertex3D
    distance = np.linalg.norm(direction)
    return vertex3D + direction * min(distance, maxVertexDisplacement) / distance

def ProcessFile(fileName):
    events = rdr.ReadRootFile(nameToPathDict[fileName])
    df = dfPfoData.query("fileName == @fileName")
    for className, classQuery in gc.classes.items():
        if className == "all":
            continue
        dfClass = df.query(classQuery)
        for index, pfoGeneralInfo in dfClass.iterrows():
            pfo = events[pfoGeneralInfo.eventId][pfoGeneralInfo.pfoId]
            maxVertexDisplacement = min((min(cc.imageSpan["U"]), min(cc.imageSpan["V"]), min(cc.imageSpan["W"]))) / 2 # Ensures vertex remains visible in all views
            centre3D = GetImage3DCentre(pfo.xCoord3D, pfo.yCoord3D, pfo.zCoord3D, pfo.vertex3D, maxVertexDisplacement)
            wireCoordFlip = -1 if pfoGeneralInfo["flip"] else 1# Mitigation for samples with non-isotropic distribution
            centre3D[1:] *= wireCoordFlip
            PlotPFOSVG(pfo.driftCoordW, pfo.wireCoordW * wireCoordFlip, pfo.driftCoordErrW, pfo.wireCoordErr, pfo.energyW, "%s/%s_%s_%s_v001.png" %(currentOutputFolder %(className), pfo.fileName, pfo.eventId, pfo.pfoId), ProjectVector(centre3D, "W"), *cc.imageSpan["W"])
            PlotPFOSVG(pfo.driftCoordU, pfo.wireCoordU * wireCoordFlip, pfo.driftCoordErrU, pfo.wireCoordErr, pfo.energyU, "%s/%s_%s_%s_v002.png" %(currentOutputFolder %(className), pfo.fileName, pfo.eventId, pfo.pfoId), ProjectVector(centre3D, "U"), *cc.imageSpan["U"])
            PlotPFOSVG(pfo.driftCoordV, pfo.wireCoordV * wireCoordFlip, pfo.driftCoordErrV, pfo.wireCoordErr, pfo.energyV, "%s/%s_%s_%s_v003.png" %(currentOutputFolder %(className), pfo.fileName, pfo.eventId, pfo.pfoId), ProjectVector(centre3D, "V"), *cc.imageSpan["V"])

def ProcessSample(dfPfoData, nameToPathDict, sampleName):
    fileNames = list(nameToPathDict.keys() & set(dfPfoData["fileName"]))
    global currentOutputFolder
    currentOutputFolder = cc.outputFolder + "/%s/" + sampleName
    if fileNames:
        with cf.ProcessPoolExecutor() as executor:
            list(tqdm(executor.map(ProcessFile, fileNames), total=len(fileNames)))
    else:
        print('No ROOT files found for ' + sampleName + ' sample!')

if __name__ == "__main__":
    EnsureFilePath(cc.outputFolder)
    for className in gc.classNames:
        EnsureFilePath(cc.outputFolder + "/" + className)
        EnsureFilePath(cc.outputFolder + "/" + className + "/train")
        EnsureFilePath(cc.outputFolder + "/" + className + "/test")

    filePaths = sorted(glob.glob(cc.rootFileDirectory + '/**/*.root', recursive=True))
    nameToPathDict = FileNameToFilePath(filePaths)
    ds.LoadPfoData([]) # Load just the data in GeneralInfo

    # Training set
    dfPfoData = ds.GetFilteredPfoData("training", "all", "training", "union")
    np.random.seed(gc.random_state)
    dfPfoData["flip"] = np.random.rand(len(dfPfoData)) < 0.5
    ProcessSample(dfPfoData, nameToPathDict, "train")

    # Testing set
    dfPfoData = ds.GetFilteredPfoData("performance", "all", "performance", "union")
    np.random.seed(gc.random_state)
    dfPfoData["flip"] = np.random.rand(len(dfPfoData)) < 0.5
    ProcessSample(dfPfoData, nameToPathDict, "test")
