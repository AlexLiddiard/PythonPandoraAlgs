import math as m

# Angular Span
angularSpan = {
    "hitFraction": 0.7,
}

# Bragg Peak
braggPeak = {
    "startFraction": 0.65,
    "endFraction": 0.15,
}

# Chain Creation
chainCreation = {
    "squareSideLength": 5,
    "localCorrelationPoints": 5,

    # Currently unused
    "rectWidth": 10,
    "rectHeight": 2.5,
    "rectOffsetX": 2.5,
    "rectOffestY": 0,
    "cubeSideLength": 5,
}

# Hit Binning
hitBinning = {
    "binWidth": 1,
    "minBins": 3,
    "maxAngleFromAxis": m.pi,
    "hitFraction": 1,
}

# Moliere Radius
moliereRadius = {
    "fraction": 0.4,
}
