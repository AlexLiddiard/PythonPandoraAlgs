import numpy as np
import matplotlib.pyplot as plt

plotStyle = {
    'font.size': 15,
    'legend.fontsize': 'large',
    'figure.figsize': (13, 10),
    'axes.labelsize': 'xx-large',
    'axes.titlesize':'xx-large',
    'xtick.labelsize':'xx-large',
    'ytick.labelsize':'xx-large',
}
plt.rcParams.update(plotStyle)

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

def CreateHistogramWire(df, histogram):
    fig, ax = plt.subplots(figsize=(10, 7.5))
    fig.tight_layout()
    ax.set_xlabel(histogram['name'])
    ax.set_xlim((histogram['bins'][0], histogram['bins'][-1]))
    ax.set_ylabel("Probability")
    if 'yAxis' in histogram:
        ax.set_yscale(histogram['yAxis'])

    for (name, filter, normFilter, fill) in histogram['filters']:
        filteredDf = df.query(filter)
        binCounts, binEdges = np.histogram(filteredDf[histogram['name']], bins = histogram['bins'])
        if normFilter != "":
            normedBinCounts = binCounts/len(df.query(normFilter))
        else:
            normedBinCounts = binCounts/len(filteredDf)
        normedBinCountsYcoord = np.concatenate(([0], np.repeat(normedBinCounts, 2), [0]))
        normedBinCountsXcoord = np.repeat(binEdges, 2)
        ax.plot(normedBinCountsXcoord, normedBinCountsYcoord, label=name)
        if fill:
            ax.fill(normedBinCountsXcoord, normedBinCountsYcoord, alpha=0.2)

    plt.tight_layout()
    plt.legend(loc='upper right', framealpha=0.5)
    return fig, ax
