import numpy as np
import matplotlib.pyplot as plt
import BaseConfig as bc
import HistoSynthesisConfig as cfg

plt.rcParams.update(cfg.plotStyle)

# Histogram Creator Programme

def CreateHistogram(df, histogram):
    fig, subplots = plt.subplots(1, len(histogram['filters']), sharey=True, figsize=(20, 7.5))

    for ax, (filter, name) in zip(subplots, histogram['filters']):
        filteredDf = df.query(filter)

        hist = np.histogram(filteredDf[histogram['name']], bins = histogram['bins'])

        normedFeatureBinCounts = hist[0]/len(filteredDf)
        binWidth = histogram['bins'][1] - histogram['bins'][0]
        barPositions = histogram['bins'][:-1] + binWidth/2

        ax.bar(barPositions, normedFeatureBinCounts, binWidth)
        ax.set_title("%s - %s" %(histogram['name'], name))
        ax.set_xlabel(histogram['name'])
        ax.set_ylabel("Probability")
        if 'yAxis' in histogram:
            ax.set_yscale(histogram['yAxis'])
        ax.yaxis.set_tick_params(which='both', labelbottom=True)
    plt.show()

def CreateHistogramWire(ax, df, histogram):
    yLabel = "Probability"
    for (name, filter, normFilter, fill) in histogram['filters']:
        dfTemp = df
        if filter != "":
            dfTemp = dfTemp.query(filter)
        binCounts, binEdges = np.histogram(dfTemp.eval(histogram['name']), bins = histogram['bins'])
        if normFilter == "":
            normedBinCounts = binCounts/len(dfTemp)
        elif normFilter == "count":
            normedBinCounts = binCounts
            yLabel = "Count"
        else:
            normedBinCounts = binCounts/len(df.query(normFilter))
        WireBarPlot(ax, normedBinCounts, binEdges, fill=fill, label=name)
        
    ax.set_ylabel(yLabel)
    ax.set_xlabel(histogram['name'])
    ax.set_xlim((histogram['bins'][0], histogram['bins'][-1]))
    if 'yAxis' in histogram:
        ax.set_yscale(histogram['yAxis'])
    ax.legend(loc='upper right', framealpha=0.5)

def WireBarPlot(ax, heights, edges, heightErrors=None, colour=None, fill=False, label=None):
    x = np.repeat(edges, 2)
    y = np.concatenate(([0], np.repeat(heights, 2), [0]))
    ax.plot(x[1:-1], y[1:-1], label=label, color=colour)
    if fill:
        ax.fill(x, y, alpha=0.2)
    if heightErrors is not None:
        x = (edges[:-1] + edges[1:]) / 2
        ax.errorbar(x, heights, yerr=heightErrors, fmt="none", capsize=2, color=colour)

