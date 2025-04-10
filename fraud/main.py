import os
import time
import datetime

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    def fraudShare(s):
        """
        Compute share of fraudulent cases
        :param s: 2D series
        :return: Percentage of fraudulent cases out of total number of instances
        """
        if s.size == 2:
            # Compute number of fraudulent cases as a percentage of total cases
            s = s.loc[1] / (s.loc[0] + s.loc[1])
        # Account for the case where all cases corresponding to a particular year are fraudulent (non-fraudulent)
        else:
            # All cases are fraudulent
            if s.keys() == 1:
                s = 1
            # All cases are non-fraudulent
            else:
                s = 0
        return s

    def plotCatData(df):
        # Plot the percentage share of fraudulent parameters by category of the merchant
        # Category
        cat = sorted(df['category'].unique())
        # Create dictionary to store percentage of fraudulent cases per category. Plot the graph thereafter
        catDict = dict.fromkeys(cat)
        for cat_ in cat:
            # Get the rows of the dataframe which correspond to the respective category
            a = df[df["category"].str.contains(cat_)]
            # Get the number of fraudulent and non-fraudulent cases
            a = a['is_fraud'].value_counts()
            # Assign value to respective key in dictionary
            catDict[cat_] = fraudShare(a)
        del a
        # Plot bar chart of fraudulent cases per category
        x, y = zip(*sorted(catDict.items()))
        plt.bar(x, y)
        plt.ylabel(ylabel="Percentage")
        plt.grid(True, which="both", ls=":")
        plt.xticks(rotation=90)
        plt.title('Percentage of Fraudulent Cases by Category')
        plt.savefig('category.png', bbox_inches="tight", dpi=dpi)
        plt.show()


    def plotAgeData(df):
        # TODO Plot the percentage share of fraudulent parameters by age of credit card holder
        # Extract the earliest to most recent date of birth in YYYY-MM-DD format
        dob = sorted(df['dob'].unique())
        # Extract only the years
        dob = list(set([int(_[:4]) for _ in dob]))
        # TODO Plot bar chart of the absolute number of users in each age group
        # Create dictionary to store percentage of fraudulent cases per year of DOB. Plot the graph thereafter
        dobDict = dict.fromkeys(dob)
        for dob_ in dob:
            # Get the rows of the dataframe which correspond to the respective category
            a = df[df["dob"].str.contains(str(dob_))]
            # Get the number of fraudulent and non-fraudulent cases
            a = a['is_fraud'].value_counts()
            # Assign value to respective key in dictionary
            dobDict[dob_] = fraudShare(a)
        del a
        # Plot bar chart of fraudulent cases per category
        x, y = zip(*sorted(dobDict.items()))
        plt.bar(x, y)
        plt.ylabel(ylabel="Percentage")
        plt.grid(True, which="both", ls=":")
        plt.xticks()
        plt.title('Percentage of Fraudulent Cases by DOB')
        plt.savefig('dob.png', bbox_inches="tight", dpi=dpi)
        plt.show()


    def plotStateData(df):
        # TODO Plot bar chart based on the state which credit card holder resides in
        # Plot the percentage share of fraudulent parameters by the state in which the credit card users reside in
        # State
        state = sorted(df['category'].unique())
        # Create dictionary to store percentage of fraudulent cases per state. Plot the graph thereafter
        stateDict = dict.fromkeys(state)
        for cat_ in state:
            # Get the rows of the dataframe which correspond to the respective state
            a = df[df["category"].str.contains(cat_)]
            # Get the number of fraudulent and non-fraudulent cases
            a = a['is_fraud'].value_counts()
            # Assign value to respective key in dictionary
            stateDict[cat_] = fraudShare(a)
        del a
        # Plot bar chart of fraudulent cases per state
        x, y = zip(*sorted(stateDict.items()))
        plt.bar(x, y)
        plt.ylabel(ylabel="Percentage")
        plt.grid(True, which="both", ls=":")
        plt.xticks()
        plt.title('Percentage of Fraudulent Cases by State')
        plt.savefig('state.png', bbox_inches="tight", dpi=dpi)
        plt.show()


    def knnFun():
        # TODO Implement the k nearest neighbour algorithm
        pass


    # -------------------------------------------Start of preamble -----------------------------------------------------
    rng = np.random.default_rng(123542871981236)
    dpi = 96
    plt.figure(figsize=(1500 / dpi, 750 / dpi), dpi=dpi)

    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    # Shuffle the datasets while loading it
    # TODO Normalise data sets
    trainSet = pd.read_csv(os.path.join(THIS_FOLDER, 'data/fraudTrain.csv')).sample(frac=1)
    testSet = pd.read_csv(os.path.join(THIS_FOLDER, 'data/fraudTest.csv')).sample(frac=1)
    # Concatenated training and test set
    concatSet = pd.concat([trainSet, testSet], axis=0)

    graphName = 'graph'
    plotFromCsv = False

    # TODO Use clustering algorithm to detect fraud
    if plotFromCsv is False:
        tic = time.time()

        # plotCatData(concatSet)
        plotAgeData(concatSet)
        # plotStateData(concatSet)
        # knnFun()

        toc = time.time()
        print("All simulations completed. Program terminating. Total time taken was",
              str(datetime.timedelta(seconds=toc - tic)))
    else:
        pass
    pass
