# This module is for track/shower algorithm #0
import math
import numpy as np

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


def GetFeature(pfo):
    return RSquared(pfo.driftCoordW, pfo.wireCoordW)
