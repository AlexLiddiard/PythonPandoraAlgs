import svgwrite as sw
import UpRootFileReader as rdr
import glob
import os
import random as rnd
import concurrent.futures as cf
import importlib
from tqdm import tqdm
import numpy as np
from UpRootFileReader import MicroBooneGeo
import CNNConfig as cc
import GeneralConfig as gc
from GetFeatureData import ProcessEvents

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
    dwg = sw.Drawing(filename=fileName, size = (256, 256), viewBox = "%s %s %s %s" %(meanDriftCoord - halfDriftSpan, meanWireCoord - halfWireSpan, driftSpan, wireSpan), debug=True)
    dwg.add(dwg.rect(insert=(meanDriftCoord - halfDriftSpan, meanWireCoord - halfWireSpan), size=(driftSpan, wireSpan), rx=None, ry=None, fill='rgb(0,0,0)'))
    for driftCoord, wireCoord, driftCoordError, energy in zip(driftCoords, wireCoords, driftCoordErrors, energies):
        ellipse = dwg.ellipse(center=(driftCoord, wireCoord), r=(driftCoordError, wireCoordError))
        ellipse.fill('white', opacity = min(energy/(wireCoordError * driftCoordError * cc.maxEnergyDensity) , 1)) #min(np.log1p(energy)/cc.logMaxEnergy, 1)
        dwg.add(ellipse)
    dwg.save()

def ProcessFile(filePath):
    events = rdr.ReadRootFile(filePath)
    df = ProcessEvents(events, [importlib.import_module("GeneralInfo")])[0]
    df = df.query(cc.preRequisites["training"])
    for className, classQuery in gc.classes.items():
        if className == "all":
            continue
        dfClass = df.query(classQuery)
        for index, pfoGeneralInfo in dfClass.iterrows():
            pfo = events[pfoGeneralInfo.eventId][pfoGeneralInfo.pfoId]
            PlotPFOSVG(pfo.driftCoordW, pfo.wireCoordW, pfo.driftCoordErrW, pfo.wireCoordErr, pfo.energyW, "%s/%s/%s_%s_%s_v001.png" %(cc.outputFolder, className, pfo.fileName, pfo.eventId, pfo.pfoId), *cc.imageSpan["W"])
            PlotPFOSVG(pfo.driftCoordU, pfo.wireCoordU, pfo.driftCoordErrU, pfo.wireCoordErr, pfo.energyU, "%s/%s/%s_%s_%s_v002.png" %(cc.outputFolder, className, pfo.fileName, pfo.eventId, pfo.pfoId), *cc.imageSpan["U"])
            PlotPFOSVG(pfo.driftCoordV, pfo.wireCoordV, pfo.driftCoordErrV, pfo.wireCoordErr, pfo.energyV, "%s/%s/%s_%s_%s_v003.png" %(cc.outputFolder, className, pfo.fileName, pfo.eventId, pfo.pfoId), *cc.imageSpan["V"])

for className in gc.classNames:
    if not os.path.exists(cc.outputFolder + "/" + className):
        os.mkdir(cc.outputFolder + "/" + className)

filePaths = glob.glob(cc.rootFileDirectory + '/**/*.root', recursive=True)
if filePaths:
    with cf.ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(ProcessFile, filePaths), total=len(filePaths)))
else:
    print('No ROOT files found!')