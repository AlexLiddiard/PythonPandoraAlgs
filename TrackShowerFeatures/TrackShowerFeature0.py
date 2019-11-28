# This module is for track/shower algorithm #0
import math
import numpy as np
import scipy.optimize as opt

# Square of the Pearson product-moment correlation
# https://en.wikipedia.org/wiki/Residual_sum_of_squares
# It is a normalised measure of the sum of the residual squares
# (of the least squares regression line).
def RSquared(xCoords, yCoords):
    Sxy = 0
    Sxx = 0
    Syy = 0
    Sx = 0
    Sy = 0
    n = xCoords.size
    if n == 0:
        return -1

    Sxy = np.sum(xCoords * yCoords)
    Sxx = np.sum(xCoords * xCoords)
    Syy = np.sum(yCoords * yCoords)
    Sx = np.sum(xCoords)
    Sy = np.sum(yCoords)
    divisor = math.sqrt((n * Sxx - Sx * Sx) * (n * Syy - Sy * Sy))
    if divisor == 0:
        return -1

    r = (n * Sxy - Sy * Sx) / divisor
    return r * r

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

def LineFit(x, y, yErr):
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
    R2 = RSquared(pfo.driftCoordW, pfo.wireCoordW)

    c, m, cErr, mErr = LineFit(pfo.wireCoordW, pfo.driftCoordW, pfo.driftCoordErrW)
    angleErr = m
    return { "F0a" : R2, "F0b" : angleErr}
