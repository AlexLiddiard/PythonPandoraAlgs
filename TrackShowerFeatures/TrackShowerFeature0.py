# This module is for track/shower algorithm #0
import math
import numpy as np
import scipy.optimize as opt

# Ordinary Least Squares line fit
def OLS(xCoords, yCoords):
    n = xCoords.size
    if n == 0:
        return float("inf"), float("inf"), -1

    Sxy = np.sum(xCoords * yCoords)
    Sxx = np.sum(xCoords * xCoords)
    Syy = np.sum(yCoords * yCoords)
    Sx = np.sum(xCoords)
    Sy = np.sum(yCoords)

    tmp1 = n * Sxx - Sx * Sx
    tmp2 = n * Syy - Sy * Sy
    tmp3 = n * Sxy - Sx * Sy
    if tmp1 == 0 or tmp2 == 0:
        return float("inf"), float("inf"), -1
    r2 = (tmp3 * tmp3) / (tmp1 * tmp2)
    b = tmp3 / tmp1
    a = Sy / n - b * Sx / n
    return a, b, r2

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

def GetFeature(pfo):
    a, b, r = OLS(pfo.driftCoordW, pfo.wireCoordW)
    return { "F0a" : r * r }
