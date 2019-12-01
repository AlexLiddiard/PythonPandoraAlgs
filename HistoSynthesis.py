import numpy as np
import matplotlib.pyplot as plt

# Histogram Creator Programme

def CreateHistogram(df, histogram):

    df = df.query(histogram['name'] + '!=-1')

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
        ax.yaxis.set_tick_params(which='both', labelbottom=True)

    plt.show()


