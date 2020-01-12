# This module is for track/shower algorithm #0
import math
import numpy as np
import scipy.optimize as opt

# Ordinary Least Squares line fit
def OLS(xCoords, yCoords):
    n = xCoords.size
    if n < 2:
        return float("inf"), float("inf"), -1
    Sxy = np.sum(xCoords * yCoords)
    Sxx = np.sum(xCoords * xCoords)
    Syy = np.sum(yCoords * yCoords)
    Sx = np.sum(xCoords)
    Sy = np.sum(yCoords)
    xvar = n * Sxx - Sx * Sx # This is actually xvar * n * n
    yvar = n * Syy - Sy * Sy # This is actually yvar * n * n
    cvar = n * Sxy - Sx * Sy # This is actually cvar * n * n
    if xvar == 0:
        return float("inf"), xCoords[0], 1
    elif yvar == 0:
        return 0, yCoords[0], 1
    else:
        b = cvar / xvar
        a = Sy / n - b * Sx / n
        r2 = (cvar * cvar) / (xvar * yvar)
        return a, b, r2

def OLSNoIntercept(xCoords, yCoords):
    Sxy = np.sum(xCoords * yCoords)
    Sxx = np.sum(xCoords * xCoords)
    if Sxx == 0:
        return float("inf")
    b = Sxy / Sxx
    return b

def RSquared(xCoords, yCoords):
    n = xCoords.size
    if n == 0:
        return -1

    Sxy = np.sum(xCoords * yCoords)
    Sxx = np.sum(xCoords * xCoords)
    Syy = np.sum(yCoords * yCoords)
    Sx = np.sum(xCoords)
    Sy = np.sum(yCoords)

    tmp1 = n * Sxy - Sx * Sy
    tmp2 = (n * Syy - Sy * Sy) * (n * Sxx - Sx * Sx)
    if tmp2 == 0:
        return -1

    return (tmp1 * tmp1) / tmp2


def LineEqn(x, a, b):
    return a * x + b

def QuadEqn(x, a, b, c):
    return a * x * x + b * x + c

def RSquaredNew(y, yFit):
    yAvg = np.mean(y)
    ss_res = np.sum((y - yFit) ** 2)
    ss_tot = np.sum((y - yAvg) ** 2)
    return 1 - (ss_res / ss_tot)

def QuadFitScipy(x, y, yErr):
    p0 = (0, 0, 0)
    popt, pcov = opt.curve_fit(QuadEqn, x, y, p0, sigma=yErr)
    yfit = QuadEqn(x, *popt)
    return RSquaredNew(y, yfit)

def LineFitScipy(x, y, yErr):
    p0 = (0, 0)
    popt, pcov = opt.curve_fit(LineEqn, x, y, p0, sigma=yErr)
    yfit = LineEqn(x, *popt)
    gradOpt = popt[0]
    gradErr = math.sqrt(pcov[0,0])
    angleErr = abs(math.atan(gradOpt + gradErr) - math.atan(gradOpt))
    return angleErr, RSquaredNew(y, yfit)

def LineFitWithError(x, y, yErr):
    S = np.sum(1/(yErr * yErr))
    Sx = np.sum(x / (yErr * yErr))
    Sy = np.sum(y / (yErr * yErr))
    Sxx = np.sum((x * x) / (yErr * yErr))
    Sxy = np.sum((x * y) / (yErr * yErr))
    delta = S * Sxx - Sx * Sx
    c = (Sxx * Sy - Sx * Sxy) / delta
    m = (S * Sxy - Sx * Sy) / delta
    cErr = math.sqrt(Sxx / delta)
    mErr = math.sqrt(S / delta)
    return c, m, cErr, mErr

def GetFeatures(pfo, calculateViews):
    featureDict = {}
    if calculateViews["U"]:
        a, b, r2 = OLS(pfo.driftCoordU, pfo.wireCoordU)
        featureDict.update({ "RSquaredU" : r2 })
    if calculateViews["V"]:
        a, b, r2 = OLS(pfo.driftCoordV, pfo.wireCoordV)
        featureDict.update({ "RSquaredV" : r2 })
    if calculateViews["W"]:
        a, b, r2 = OLS(pfo.driftCoordW, pfo.wireCoordW)
        featureDict.update({ "RSquaredW" : r2 })
    return featureDict
