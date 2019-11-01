# This module is for track/shower algorithm #2
import statistics
import math

# Algorithm parameters
#maxSeparationU = 2
chainThresholdU = 7
rectWidthU = 1.5
rectHeightU = 1.5
crumpleConstU = 1

#maxSeparationV = 2
chainThresholdV = 7
rectWidthV = 1.5
rectHeightV = 1.5
crumpleConstV = 1

#maxSeparationW = 2
chainThresholdW = 0.5
rectWidthW = 5
rectHeightW = 5
crumpleConstW = 5

#maxSeparationU2 = maxSeparationU * maxSeparationU
#maxSeparationV2 = maxSeparationV * maxSeparationV
#maxSeparationW2 = maxSeparationW * maxSeparationW

# Finds the nearest neighbour of a 2D point (out of a list of points)
# It's quite slow
def NearestPoint(pointX, pointY, pointListX, pointListY):
	nearestPointIndex = 0
	shortestDistance2 = float("inf")
	for i in range(0, len(pointListX)):
		tmp = Distance2(pointX, pointY, pointListX[i], pointListY[i])
		if tmp < shortestDistance2:
			nearestPointIndex = i
			shortestDistance2 = tmp
	return nearestPointIndex, shortestDistance2

# Finds the nearest neighbour of a 2D point (out of a list of points)
# The list of points checked is a subsample of the input list. It consists of all points within a rectangle
# of width rectWidth and height rectHeight centred on the point.
# If the subsample is empty, a distance of infinity and point index of -1 is returned.
def NearestPointInRectangle(pointX, pointY, pointListX, pointListY, rectWidth, rectHeight):
	xLow = pointX - rectWidth / 2
	xHigh = pointX + rectWidth / 2
	yLow = pointY - rectHeight / 2
	yHigh = pointY + rectHeight / 2
	nearestPointIndex = -1
	shortestDistance2 = float("inf")
	for i in range(0, len(pointListX)):
		xTest = pointListX[i]
		yTest = pointListY[i]
		if xTest < xLow:
			continue
		if xTest > xHigh:
			continue
		if yTest < yLow:
			continue
		if yTest > yHigh:
			continue
		
		distance2 = Distance2(pointX, pointY, pointListX[i], pointListY[i])
		if distance2 < shortestDistance2:
			nearestPointIndex = i
			shortestDistance2 = distance2
	return nearestPointIndex, shortestDistance2

# Resurns the square of the separation distance of a pair of 2D points
def Distance2(pointAX, pointAY, pointBX, pointBY):
	deltaX = pointAX - pointBX
	deltaY = pointAY - pointBY
	return deltaX * deltaX + deltaY * deltaY

# Creates a list of points that form a chain. Each point pair in the chan has a squared separation smaller than maxSeparation2.
# The chain starts with the first point in the input list
# The chain ends when no more nearby points can be found.
# The points added to the chain are removed from the input list of points.
def CreatePointChain(pointListX, pointListY, maxSeparation2):
	currentX = pointListX.pop(0)
	currentY = pointListY.pop(0)
	chainX = [currentX]
	chainY = [currentY]
	nearbyPoints = True
	while (nearbyPoints):
		nearestPointIndex = NearestPoint(currentX, currentY, pointListX, pointListY)
		if distance < maxSeparation2:
			currentX = pointListX.pop(nearestPointIndex)
			currentY = pointListY.pop(nearestPointIndex)
			chainX.append(currentX)
			chainY.append(currentY)
		else:
			nearbyPoints = False
	return chainX, chainY

def CreatePointChain2(pointListX, pointListY, rectWidth, rectHeight):
	currentX = pointListX.pop(0)
	currentY = pointListY.pop(0)
	chainX = [currentX]
	chainY = [currentY]
	chainLength = 0
	nearbyPoints = True
	while (nearbyPoints):
		nearestPointIndex, distance2 = NearestPointInRectangle(currentX, currentY, pointListX, pointListY, rectWidth, rectHeight)
		if nearestPointIndex >= 0:
			currentX = pointListX.pop(nearestPointIndex)
			currentY = pointListY.pop(nearestPointIndex)
			chainX.append(currentX)
			chainY.append(currentY)
			chainLength += math.sqrt(distance2)
		else:
			nearbyPoints = False
	return chainX, chainY, chainLength

def ShowerInView(driftCoord, wireCoord, chainThreshold, rectWidth, rectHeight, crumpleConst):		
	chainScore = 0
	while driftCoord:	# While pfo hit list is not empty
		chainX, chainY, chainLength = CreatePointChain2(driftCoord, wireCoord, rectWidth, rectHeight)
		if len(chainX) > 1:
			crumpleFactor = math.sqrt(Distance2(chainX[0], chainY[0], chainX[-1], chainY[-1])) / chainLength
		else:
			crumpleFactor = 1
		chainScore += 1 - crumpleFactor

	if (chainScore > chainThreshold):
		return 1, chainScore
	else:
		return 0, chainScore

def RunAlgorithm(pfo):
	# Using only W view
	showerInViewW = ShowerInView(pfo.driftCoordW, pfo.wireCoordW, chainThresholdW, rectWidthW, rectHeightW, crumpleConstW)
	print("ChainScore: %.2f" % showerInViewW[1], end = " ")
	return showerInViewW[0]
	
	# Using all three views (slower)
'''
	showerStats = []
	showerInViewU = ShowerInView(pfo.driftCoordU, pfo.wireCoordU, chainCountThresholdU, maxSeparationU2)
	showerInViewV = ShowerInView(pfo.driftCoordV, pfo.wireCoordV, chainCountThresholdV, maxSeparationV2)
	showerInViewW = ShowerInView(pfo.driftCoordW, pfo.wireCoordW, chainCountThresholdW, maxSeparationW2)
	print("ChainCounts: %d %d %d" % (showerInViewU[1], showerInViewV[1], showerInViewW[1]), end = " ")

	showerStats = (showerInViewU[0], showerInViewV[0], showerInViewW[0])
	if (showerStats.count(1) > 1): 					# A good shower score
		return 1
	elif (showerStats.count(0) > 1): 				# A good track score
		return 0
	elif showerStats.count(-1) == 2 and showerStats[2] != -1:	# The case where only info from one plane. The W plane alone gives adequate info for a track/shower decision. 
		return showerStats[2]
	else: # Not enough info to decide
		return -1
'''
