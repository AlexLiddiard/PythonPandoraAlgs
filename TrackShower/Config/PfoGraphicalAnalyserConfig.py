import BaseConfig as bc

############################################## PFO GRAPHICAL ANALYSER CONFIGURATION ##################################################

# The folder to read ROOT files from
rootFileDirectory = bc.pythonFolderFull + "/ROOT Files"

# The data sample to select PFOs from
# If unused, PFOs will be opened directly from ROOT files without any pre-fitering applied
useDataSample = True
randomisePfos = True
dataSample = {
    "dataSource": "performance",
    "pfoClass": "all",
    "filterClass": "performance",
    "filterName": "union"
}

# Additional filters to be applied before selecting a PFO
filters = [
    #### PFO selection ####
    #"isShower==0 and Likelihood>0.001 and abs(mcpEnergy - 1.05) < 0.05",
    #"mcPdgCode==2212 and mcpMomentum < 1",
    #'AngularSpanW > 2',
    #'Likelihood > 0.89 and nHitsW > 200 and isShower != 1', # shower-like muons/protons/etc. with many hits
    #'likelihood > 0.89 and absPdgCode==2212' # shower-like protons
    #'likelihood < 0.89 and absPdgCode==11' # track-like electrons
    #'BinnedHitStdW==0' # BinnedHitStd anomaly
    #"completenessW==0.5",
    #"completenessV==0",
    #'nHitsV <= 59',
    #'nHitsV >= 51',
    #'mcPdgCode == 22',
    #'BDTV > 1.088',
    #'mcHierarchyTier == 2',
    #'mcParentPdgCode == 111',
    'abs(mcPdgCode) == 13',
    'ChainRSquaredAvgW < 0.25',
    'ChainRSquaredAvgW > 0.20',
    'purityW > 0.95',
    'completenessW > 0.95',
]

# Info to be displayed beside the PFO
additionalInfo = [
    'mcNuanceCode',
    'mcHierarchyTier',
    'mcpMomentum',
    'mcNuMomentum',
    'mcParentPdgCode',
    'mcNuPdgCode',
    #'BinnedHitStdU',
    #'ChainRatioAvgU',
    #'ChainRSquaredStdU',
    #'AngularSpanU',
    #'BinnedHitStdV',
    #'ChainRatioAvgV',
    #'ChainRSquaredStdV',
    #'AngularSpanV',
    #'BinnedHitStdW',
    #'ChainRatioAvgW',
    #'ChainRSquaredStdW',
    #'AngularSpanW',
    #'Likelihood',
]

# The view to be displayed: U, V, or W
view = "W"

showTitle = True
plotStyle = {
    'font.size': 15,
    'legend.fontsize': 'xx-large',
    'figure.figsize': (13,10),
    'axes.labelsize': 'xx-large',
    'axes.titlesize':'xx-large',
    'xtick.labelsize':'xx-large',
    'ytick.labelsize':'xx-large'
}