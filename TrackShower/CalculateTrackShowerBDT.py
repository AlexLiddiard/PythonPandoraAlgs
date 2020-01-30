import BDTCalculator as bdt
import DataSampler as ds
import TrackShowerSampling as tss

# Load the training PFOs
ds.LoadPfoData(tss.dataFolder, tss.allDataSources, features)

dfTrainingPfoData = ds.GetFilteredPfoData(tss.trainingDataSources, tss.classQueries, tss.trainingPreFilters, "all", #ds.dfTrainingPfoData["all"]
dfBdtValues = bdt.GetAllBDTData(dfTrainingPfoData, ds.dfInputPfoData, ds.GetFeatureViews(features), "isShower==0")

print("\nSaving results")
ds.SavePfoData(dfBdtValues, "DecisionTreeCalculator")
print("Finished!")