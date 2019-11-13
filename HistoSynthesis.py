import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import pandas as pd

filename = 'featureData.pickle'

#Histogram Creator Programme

def GetFeatureStats(featureName, df_shower, df_track, bins, ymax):
    fig = plt.figure(figsize=(20,7.5))
    ax1 = fig.add_subplot(1,3,1)
    ax2 = fig.add_subplot(1,3,2)
    ax3 = fig.add_subplot(1,3,3)
    shower_filter = df_shower[featureName] != -1
    track_filter = df_track[featureName] != -1
    showerFeatureData = df_shower[shower_filter][featureName]
    trackFeatureData = df_track[track_filter][featureName]
    ax1.hist(showerFeatureData, bins=bins, density=1)
    ax2.hist(trackFeatureData, bins=bins, density=1)
    ax3.hist(showerFeatureData, bins=bins, density=1)
    ax3.hist(trackFeatureData, bins=bins, density=1)

    ax1.set_ylim([0,ymax])
    ax2.set_ylim([0,ymax])
    ax3.set_ylim([0,ymax])
    plt.show()

    shower_hist = np.histogram(showerFeatureData, bins=bins)
    track_hist = np.histogram(trackFeatureData, bins=bins)
    fS = st.rv_histogram(shower_hist).pdf
    fT = st.rv_histogram(track_hist).pdf
    return fS, fT


def L(featurePdfPairs, featureValues):
    Pt = 1
    Ps = 1
    for i in range(0, len(featureValues)):
        if featureValues[i] == -1:
            continue
        Pt *= featurePdfPairs[i][0](featureValues[i])
        Ps *= featurePdfPairs[i][1](featureValues[i])
    return Ps / (Pt + Ps)

'''Separate true tracks from true showers. Then plot histograms for feature
values. Convert these histograms in to PDFs using scipy. Plot overlapping
histograms for track and shower types.'''

# Load the pickle file.
df = pd.read_pickle(filename)
is_shower = df['pfoTrueType'] == 1
is_track = df['pfoTrueType'] == 0
df_shower = df[is_shower]
df_track = df[is_track]

# Make histograms and PDFs
featurePdfPairs = [GetFeatureStats("F0a", df_shower, df_track, np.linspace(0, 1, num=200), 40),
                   GetFeatureStats("F1a", df_shower, df_track, np.linspace(0, 6, num=200), 3),
                   GetFeatureStats("F2a", df_shower, df_track, np.linspace(0, 30, num=31), 1),
                   GetFeatureStats("F2b", df_shower, df_track, np.linspace(0, 1, num=200), 40),
                   GetFeatureStats("F2c", df_shower, df_track, np.linspace(0, 1, num=200), 40)]


# All features combined
showerFeatureValues = df_shower[['F0a','F1a','F2a','F2b','F2c']].to_numpy()
trackFeatureValues = df_track[['F0a','F1a','F2a','F2b','F2c']].to_numpy()
nShowers = len(showerFeatureValues)
nTracks = len(trackFeatureValues)
Lshowers = np.zeros(nShowers)
Ltracks = np.zeros(nShowers)
for i in range(0, nShowers):
    Lshowers[i] = L(featurePdfPairs, showerFeatureValues[i])
for i in range(0, nShowers):
    Ltracks[i] = L(featurePdfPairs, trackFeatureValues[i])

fig = plt.figure(figsize=(20,7.5))
ax1 = fig.add_subplot(1,3,1)
ax2 = fig.add_subplot(1,3,2)
ax3 = fig.add_subplot(1,3,3)
ax1.hist(Lshowers, bins=200, density=1)
ax2.hist(Ltracks, bins=200, density=1)
ax3.hist(Lshowers, bins=200, density=1)
ax3.hist(Ltracks, bins=200, density=1)
plt.show()