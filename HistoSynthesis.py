import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd

filename = '/home/jack/Documents/Pandora/PythonPandoraAlgs/featureData.pickle'

#Histogram Creator Programme

#Load the pickle file.

'''Separate true tracks from true showers. Then plot histograms for feature 
values. Convert these histograms in to PDFs using scipy. Plot overlapping 
histograms for track and shower types.'''

df = pd.read_pickle(filename)

is_shower = df['pfoTrueType'] == 1
df_shower = df[is_shower]
is_track = df['pfoTrueType'] == 0
df_track = df[is_track]

fig = plt.figure()
fig.set_dpi(200)
ax = fig.add_subplot(111)


'''
f0a_shower_filter = df_shower['F0a'] != -1
df_shower_f0a = df_shower[f0a_shower_filter]
ax.hist(df_shower_f0a["F0a"], bins=100, density=1)

f0a_track_filter = df_track['F0a'] != -1
df_track_f0a = df_track[f0a_track_filter]
ax.hist(df_track_f0a["F0a"], bins=100, density=1)
'''

'''
f1a_shower_filter = df_shower['F1a'] != -1
df_shower_f1a = df_shower[f1a_shower_filter]
ax.hist(df_shower_f1a["F1a"], bins=100, density=1)

f1a_track_filter = df_track['F1a'] != -1
df_track_f1a = df_track[f1a_track_filter]
ax.hist(df_track_f1a["F1a"], bins=100, density=1)
'''

'''
f2a_shower_filter = df_shower['F2a'] != -1
df_shower_f2a = df_shower[f2a_shower_filter]
ax.hist(df_shower_f2a["F2a"], bins=25, density=1)

f2a_track_filter = df_track['F2a'] != -1
df_track_f2a = df_track[f2a_track_filter]
ax.hist(df_track_f2a["F2a"], bins=25, density=1)
'''


f2b_shower_filter = df_shower['F2b'] != -1
df_shower_f2b = df_shower[f2b_shower_filter]
ax.hist(df_shower_f2b["F2b"], bins=100, density=1)

f2b_track_filter = df_track['F2b'] != -1
df_track_f2b = df_track[f2b_track_filter]
ax.hist(df_track_f2b["F2b"], bins=100, density=1)
