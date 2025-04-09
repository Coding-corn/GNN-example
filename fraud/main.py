import os
import time
import datetime

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    def plotData(df):
        # TODO Analyse the share of fraudulent transactions by parameter
        # Category
        cat = sorted(df['category'].unique())
        mer = sorted(df['merchant'].unique())
        # Create dictionary to store percentage of fraudulent cases per category. Plot the graph thereafter
        catDict = dict.fromkeys(cat)
        for cat_ in cat:
            # Get the rows of the dataframe which correspond to the respective category
            a = df[df["category"].str.contains(cat_)]
            # Get the number of fraudulent and non-fraudulent cases
            a = a['is_fraud'].value_counts()
            # Compute number of fraudulent cases as a percentage of total cases
            a = a.loc[1] / (a.loc[0] + a.loc[1])
            # Assign value to respective key in dictionary
            catDict[cat_] = a
        del a
        # Plot bar chart of fraudulent cases per category
        x, y = zip(*sorted(catDict.items()))
        plt.bar(x, y)
        plt.ylabel(ylabel="Percentage")
        plt.grid(True, which="both", ls=":")
        plt.xticks(rotation=90)
        plt.title('Percentage of Fraudulent Cases by Category')
        plt.savefig(str(graphName) + '.png', bbox_inches="tight", dpi=dpi)
        plt.show()
        pass


    # -------------------------------------------Start of preamble -----------------------------------------------------
    rng = np.random.default_rng(123542871981236)
    dpi = 96
    plt.figure(figsize=(1500 / dpi, 750 / dpi), dpi=dpi)

    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    # Shuffle the datasets while loading it
    trainSet = pd.read_csv(os.path.join(THIS_FOLDER, 'data/fraudTrain.csv')).sample(frac=1)
    testSet = pd.read_csv(os.path.join(THIS_FOLDER, 'data/fraudTest.csv')).sample(frac=1)

    graphName = 'graph'
    plotFromCsv = False

    # TODO Use clustering algorithm to detect fraud
    if plotFromCsv is False:
        tic = time.time()

        plotData(pd.concat([trainSet, testSet], axis=0))

        toc = time.time()
        print("All simulations completed. Program terminating. Total time taken was",
              str(datetime.timedelta(seconds=toc - tic)))
    else:
        pass
    pass
