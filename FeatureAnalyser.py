import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import math as m
from UpRootFileReader import MicroBooneGeo
from HistoSynthesis import CreateHistogramWire
import DataSampler as ds

def GetFeatureView(featureName):
    if featureName.endswith("3D"):
        return "3D"
    if featureName[-1] in ["U", "V", "W"]:
        return featureName[-1]
    else:
        return "intersection"

def Purity(efficiency1, efficiency2, n1, n2):
    Purity1 = 1/(1 + (1 - efficiency2)*n2/(efficiency1*n1))
    return Purity1

def PurityError(efficiency1, efficiencyError1, efficiency2, efficiencyError2, n1, n2):
    purity1 = Purity(efficiency1, efficiency2, n1, n2)
    purityOff1a = Purity(efficiency1 + efficiencyError1, efficiency2, n1, n2)
    purityOff1b = Purity(efficiency1, efficiency2 + efficiencyError2, n1, n2)
    purityErr1a = purityOff1a - purity1
    purityErr1b = purityOff1b - purity1
    purityErr1 = m.sqrt(purityErr1a**2 + purityErr1b**2)
    return purityErr1

def PurityEfficiencyError(purity, purityError, efficiency, efficiencyError):
    return m.sqrt((purityError*efficiency)**2 + (purity*efficiencyError)**2)

def PurityEfficiency(dfClass0Variable, dfClass1Variable, cutOff, class1CutDirection):
    sumClass0 = len(dfClass0Variable)
    sumClass1 = len(dfClass1Variable)
    sumCorrectClass1 = (dfClass1Variable > cutOff).sum()
    sumIncorrectClass1 = (dfClass1Variable < cutOff).sum()
    sumCorrectClass0 = (dfClass0Variable < cutOff).sum()
    sumIncorrectClass0 = (dfClass0Variable > cutOff).sum()
    if class1CutDirection == "left":
        sumCorrectClass1, sumIncorrectClass1 = sumIncorrectClass1, sumCorrectClass1
        sumCorrectClass0, sumIncorrectClass0 = sumIncorrectClass0, sumCorrectClass0
    class0Efficiency = sumCorrectClass0/(sumCorrectClass0+sumIncorrectClass0)
    class0Purity = sumCorrectClass0/(sumCorrectClass0 + sumIncorrectClass1)
    class0PurityEfficiency = class0Purity * class0Efficiency
    class1Efficiency = sumCorrectClass1/(sumCorrectClass1+sumIncorrectClass1)
    class1Purity = sumCorrectClass1/(sumCorrectClass1 + sumIncorrectClass0)
    class1PurityEfficiency = class1Purity * class1Efficiency
    class0EfficiencyError = m.sqrt(class0Efficiency * (1 - class0Efficiency) / sumClass0)
    class1EfficiencyError = m.sqrt(class1Efficiency * (1 - class1Efficiency) / sumClass1)
    class0PurityError = PurityError(class0Efficiency, class0EfficiencyError, class1Efficiency, class1EfficiencyError, sumClass0, sumClass1)
    class1PurityError = PurityError(class1Efficiency, class1EfficiencyError, class0Efficiency, class0EfficiencyError, sumClass1, sumClass0)
    class0PurityEfficiencyError = PurityEfficiencyError(class0PurityEfficiency, class0PurityError, class0Efficiency, class0EfficiencyError)
    class1PurityEfficiencyError = PurityEfficiencyError(class1PurityEfficiency, class1PurityError, class1Efficiency, class1EfficiencyError)
    return (
        class0Efficiency, class0EfficiencyError, class0Purity, class0PurityError, class0PurityEfficiency, class0PurityEfficiencyError, 
        class1Efficiency, class1EfficiencyError, class1Purity, class1PurityError, class1PurityEfficiency, class1PurityEfficiencyError
    )

def GraphCutoffLine(ax, classNames, cutoff, arrows=False, flipArrows=False):
    ax.axvline(cutoff)
    if arrows:
        if not flipArrows:
            ax.annotate(
                classNames[0], xy=(cutoff, 0.93), xycoords=ax.get_xaxis_transform(), ha="right", va="center",
                xytext=(-18,0), textcoords="offset points", bbox={'boxstyle': "larrow", 'fc': 'C1', 'ec': 'C1', 'alpha': 0.5}, fontsize=20
            )
            ax.annotate(
                classNames[1], xy=(cutoff, 0.93), xycoords=ax.get_xaxis_transform(), ha="left", va="center",
                xytext=(18,0), textcoords="offset points", bbox={'boxstyle': "rarrow", 'fc': 'C0', 'ec': 'C0', 'alpha': 0.5}, fontsize=20
            )
        else:
            ax.annotate(
                classNames[0], xy=(cutoff, 0.93), xycoords=ax.get_xaxis_transform(), ha="left", va="center",
                xytext=(18,0), textcoords="offset points", bbox={'boxstyle': "rarrow", 'fc': 'C1', 'ec': 'C1', 'alpha': 0.5}, fontsize=20
            )
            ax.annotate(
                classNames[1], xy=(cutoff, 0.93), xycoords=ax.get_xaxis_transform(), ha="right", va="center",
                xytext=(-18,0), textcoords="offset points", bbox={'boxstyle': "larrow", 'fc': 'C0', 'ec': 'C0', 'alpha': 0.5}, fontsize=20
            )

def OptimiseCutoff(dfClass0Data, dfClass1Data, variableName, testCutoffs, class1CutDirection):
    dfClass0Variable = dfClass0Data[variableName]
    dfClass1Variable =  dfClass1Data[variableName]
    class0Efficiencies = []
    class0Purities = []
    class1Efficiencies = []
    class1Purities = []
    class0PurityEfficiencies = []
    class1PurityEfficiencies = []
    bestClass1Cutoff = 0
    bestClass0Cutoff = 0
    bestClass0PurityEfficiency = 0
    bestClass1PurityEfficiency = 0
    for cutoff in testCutoffs:
        (
            class0Efficiency, class0EfficiencyError, class0Purity, class0PurityError, class0PurityEfficiency, class0PurityEfficiencyError, 
            class1Efficiency, class1EfficiencyError, class1Purity, class1PurityError, class1PurityEfficiency, class1PurityEfficiencyError
        ) = PurityEfficiency(dfClass0Variable, dfClass1Variable, cutoff, class1CutDirection)
        class0Efficiencies.append(class0Efficiency)
        class0Purities.append(class0Purity)
        if class0PurityEfficiency > bestClass0PurityEfficiency and cutoff not in (0, 1):
            bestClass0PurityEfficiency = class0PurityEfficiency
            bestClass0Cutoff = cutoff
        class0PurityEfficiencies.append(class0PurityEfficiency)

        class1Efficiencies.append(class1Efficiency)
        class1Purities.append(class1Purity)
        if class1PurityEfficiency > bestClass1PurityEfficiency and cutoff not in (0, 1):
            bestClass1PurityEfficiency = class1PurityEfficiency
            bestClass1Cutoff = cutoff
        class1PurityEfficiencies.append(class1PurityEfficiency)
    return (
        bestClass0Cutoff, class0Efficiencies, class0Purities, class0PurityEfficiencies, 
        bestClass1Cutoff, class1Efficiencies, class1Purities, class1PurityEfficiencies
    )

def PrintPurityEfficiency(dfClass0Data, dfClass1Data, classNames, predictorName, cutoff, cutDirection='right'):
    dfClass0Variable = dfClass0Data[predictorName]
    dfClass1Variable =  dfClass1Data[predictorName]
    print(
        (classNames[0] + " Efficiency %.3f+-%.3f\n" +
        classNames[0] + " Purity %.3f+-%.3f\n" +
        classNames[0] + " Purity * Efficiency %.3f+-%.3f\n" +
        classNames[1] + " Efficiency %.3f+-%.3f\n" +
        classNames[1] + " Purity %.3f+-%.3f\n" +
        classNames[1] + " Purity * Efficiency %.3f+-%.3f")
    % PurityEfficiency(dfClass0Variable, dfClass1Variable, cutoff, cutDirection)
    )

def PlotVariableHistogram(dfPfoData, classNames, variable, variableHistogram, bestCutoff = None):
    variable['filters'] = variableHistogram["filters"]
    variable['yAxis'] = 'log'
    if "bins" in variableHistogram:
        variable["bins"] = variableHistogram["bins"]
    fig, ax = CreateHistogramWire(dfPfoData, variable)
    if bestCutoff is not None:
        GraphCutoffLine(ax, classNames, bestCutoff, True, variable['cutDirection'] == 'left')
    plt.savefig('%s distribution for %s' % (variable['name'], ', '.join((filter[0] for filter in variable['filters'])) + '.svg'), format='svg', dpi=1200)
    plt.show()

def GetBestPurityEfficiency(dfClass0Data, dfClass1Data, classNames, variable, nTestCuts):
    cutoffValues = np.linspace(variable["bins"][0], variable["bins"][-1], nTestCuts)
    cutoffResults = OptimiseCutoff(dfClass0Data, dfClass1Data, variable['name'], cutoffValues, variable['cutDirection'])

    # Printing results for optimal purity and efficiency
    print("Performance results for %s:" % variable['name'])
    print("\nOptimal %s cutoff %.3f" % (classNames[0], cutoffResults[0]))
    PrintPurityEfficiency(dfClass0Data, dfClass1Data, classNames, variable['name'], cutoffResults[0], variable['cutDirection'])
    print("\nOptimal %s cutoff %.3f" % (classNames[1], cutoffResults[4]))
    PrintPurityEfficiency(dfClass0Data, dfClass1Data, classNames, variable['name'], cutoffResults[4], variable['cutDirection'])
    return cutoffValues, cutoffResults

def PlotPurityEfficiencyVsCutoff(featureName, classNames, cutoffValues, cutoffResults):
        fig = plt.figure(figsize=(20,7.5))
        bx1 = fig.add_subplot(1,2,1)
        bx2 = fig.add_subplot(1,2,2)
        lines = bx1.plot(cutoffValues, cutoffResults[1], 'b', cutoffValues, cutoffResults[2], 'r', cutoffValues, cutoffResults[3], 'g')
        bx1.legend(lines, ('Efficiency', 'Purity', 'Purity * Efficiency'), loc='lower center')
        lines = bx2.plot(cutoffValues, cutoffResults[5], 'b', cutoffValues, cutoffResults[6], 'r', cutoffValues, cutoffResults[7], 'g')
        bx2.legend(lines, ('Efficiency', 'Purity', 'Purity * Efficiency'), loc='lower center')
        GraphCutoffLine(bx1, classNames, cutoffResults[0])
        GraphCutoffLine(bx2, classNames, cutoffResults[4])

        bx1.set_ylim([0, 1])
        bx1.set_title("Purity/Efficiency vs %s Cutoff — %s" % (featureName, classNames[0]))
        bx1.set_xlabel(featureName + " Cutoff")
        bx1.set_ylabel("Fraction")

        bx2.set_ylim([0, 1])
        bx2.set_title("Purity/Efficiency vs %s Cutoff — %s" % (featureName, classNames[1]))
        bx2.set_xlabel(featureName + " Cutoff")
        bx2.set_ylabel("Fraction")

        plt.tight_layout()
        plt.savefig("PurityEfficiencyVs%sCutoff.svg" % featureName, format='svg', dpi=1200)
        plt.show()

def CorrelationMatrix(featureNames, viewsUsed, preFilters, pfoData):
    dataCorrFilters = [preFilters[x] for x in viewsUsed]
    dfPfoDataCorr = pfoData["all"]["general"].query("(" + ") and (".join(dataCorrFilters) + ")")
    rMatrix = dfPfoDataCorr[featureNames].corr()
    rSquaredMatrix = rMatrix * rMatrix
    sn.heatmap(rSquaredMatrix, annot=True, annot_kws={"size": 20}, cmap="Blues")
    plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
    plt.tight_layout()
    plt.savefig("FeatureRSquaredMatrix.svg", format='svg', dpi=1200)
    plt.show()