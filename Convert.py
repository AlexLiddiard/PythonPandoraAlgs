import pandas as pd

myTestArea = "/home/tomalex/Pandora"
inputPickleFile = myTestArea + '/PythonPandoraAlgs/featureData.bz2'
dfPfoData = pd.read_pickle(inputPickleFile)
dfPfoData = dfPfoData.rename(columns={
    "ChainAvgLengthRatioU": "ChainRatioAvgU", "ChainAvgAvgRSquaredU": "ChainRSquaredAvgU", "ChainStdLengthRatioU": "ChainRatioStdU", "ChainAvgStdRSquaredU": "ChainRSquaredStdU",
    "ChainAvgLengthRatioV": "ChainRatioAvgV", "ChainAvgAvgRSquaredV": "ChainRSquaredAvgV", "ChainStdLengthRatioV": "ChainRatioStdV", "ChainAvgStdRSquaredV": "ChainRSquaredStdV",
    "ChainAvgLengthRatioW": "ChainRatioAvgW", "ChainAvgAvgRSquaredW": "ChainRSquaredAvgW", "ChainStdLengthRatioW": "ChainRatioStdW", "ChainAvgStdRSquaredW": "ChainRSquaredStdW"})
print(dfPfoData.keys())

dfPfoData.to_pickle(inputPickleFile)
