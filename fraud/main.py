import os
import time
import datetime

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import geopy.distance
from concurrent.futures import ThreadPoolExecutor

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
        # TODO Rotate bar chart
        plt.figure(figsize=(1500 / dpi, 750 / dpi), dpi=dpi)
        plt.bar(x, y)
        plt.ylabel(ylabel="Percentage")
        plt.grid(True, which="both", ls=":")
        plt.xticks(rotation=90)
        plt.title('Percentage of Fraudulent Cases by Category')
        plt.savefig('category.png', bbox_inches="tight", dpi=dpi)
        plt.show()


    def plotAgeData(df):
        # Plot the percentage share of fraudulent parameters by age of credit card holder
        # Extract the earliest to most recent date of birth in YYYY-MM-DD format
        dob = sorted(df['dob'].unique())
        # Extract only the years
        dob = list(set([int(_[:4]) for _ in dob]))
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
        plt.figure(figsize=(1500 / dpi, 750 / dpi), dpi=dpi)
        plt.bar(x, y)
        plt.ylabel(ylabel="Percentage")
        plt.grid(True, which="both", ls=":")
        plt.xticks()
        plt.title('Percentage of Fraudulent Cases by DOB')
        plt.savefig('dob.png', bbox_inches="tight", dpi=dpi)
        plt.show()

        # Plot bar chart of the absolute number of users in each age group
        # Create dictionary to store percentage of fraudulent cases per year of DOB. Plot the graph thereafter
        dobDict = dict.fromkeys(dob)
        for dob_ in dob:
            # Get the rows of the dataframe which correspond to the respective category
            a = df[df["dob"].str.contains(str(dob_))]
            # Assign value to respective key in dictionary
            dobDict[dob_] = a.shape[0]
        del a
        # Plot bar chart of fraudulent cases per category
        x, y = zip(*sorted(dobDict.items()))
        plt.figure(figsize=(1500 / dpi, 750 / dpi), dpi=dpi)
        plt.bar(x, y)
        plt.ylabel(ylabel="Total number of transactions")
        plt.grid(True, which="both", ls=":")
        plt.xticks()
        plt.title('Total Number of Transactions by DOB')
        plt.savefig('dobTrans.png', bbox_inches="tight", dpi=dpi)
        plt.show()


    def plotStateData(df):
        # Plot the percentage share of fraudulent parameters by the state in which the credit card users reside in
        # State
        state = sorted(df['state'].unique())
        # Create dictionary to store percentage of fraudulent cases per state. Plot the graph thereafter
        stateDict = dict.fromkeys(state)
        for state_ in state:
            # Get the rows of the dataframe which correspond to the respective state
            a = df[df["state"].str.contains(state_)]
            # Get the number of fraudulent and non-fraudulent cases
            a = a['is_fraud'].value_counts()
            # Assign value to respective key in dictionary
            stateDict[state_] = fraudShare(a)
        del a
        # Plot bar chart of fraudulent cases per state
        x, y = zip(*sorted(stateDict.items()))
        plt.figure(figsize=(1500 / dpi, 750 / dpi), dpi=dpi)
        plt.bar(x, y)
        plt.ylabel(ylabel="Percentage")
        plt.grid(True, which="both", ls=":")
        plt.xticks()
        plt.title('Percentage of Fraudulent Cases by State')
        plt.savefig('state.png', bbox_inches="tight", dpi=dpi)
        plt.show()

        # Plot the total number of samples taken from each respective state
        stateDict = dict.fromkeys(state)
        for state_ in state:
            # Get the rows of the dataframe which correspond to the respective state
            a = df[df["state"].str.contains(state_)]
            # Assign value to respective key in dictionary
            stateDict[state_] = a.shape[0]
        del a
        # Plot bar chart of fraudulent cases per state
        x, y = zip(*sorted(stateDict.items()))
        plt.figure(figsize=(1500 / dpi, 750 / dpi), dpi=dpi)
        plt.bar(x, y)
        plt.ylabel(ylabel="Total number of transactions")
        plt.grid(True, which="both", ls=":")
        plt.xticks()
        plt.title('Total Number of Transactions by State')
        plt.savefig('stateTrans.png', bbox_inches="tight", dpi=dpi)
        plt.show()


    # TODO May have to delete this function due to potential irrelevancy
    def plotJobData(df):
        # TODO xlabel is too cluttered due to the number of unique jobs. Need to reformat the figure
        # Plot barchart based on occupation of the credit card holder
        job = sorted(df['job'].unique())
        jobDict = dict.fromkeys(job)
        for job_ in job:
            a = df[df["job"].str.contains(job_)]
            a = a['is_fraud'].value_counts()
            jobDict[job_] = fraudShare(a)
        del a
        plt.figure(figsize=(1500 / dpi, 750 / dpi), dpi=dpi)
        x, y = zip(*sorted(jobDict.items()))
        plt.bar(x, y)
        plt.ylabel(ylabel="Percentage")
        plt.grid(True, which="both", ls=":")
        plt.xticks(rotation=90)
        plt.title('Percentage of Fraudulent Cases by Job')
        plt.savefig('job.png', bbox_inches="tight", dpi=dpi)
        plt.show()


    def plotDistData(df, parallel: bool = False):
        # Function to compute geodesic distance between cardholder and merchant
        def compute_distance(i):
            return geopy.distance.geodesic(user[i], merch[i]).km

        # Plot barchart based on distance between credit card holder and merchant wrt latitude and longitude
        lat = df["lat"].tolist()
        long = df["long"].tolist()
        merch_lat = df["merch_lat"].tolist()
        merch_long = df["merch_long"].tolist()
        # Coordinate of cardholder
        user = [(lat[i], long[i]) for i in range(df.shape[0])]
        # Coordinate of merchant
        merch = [(merch_lat[i], merch_long[i]) for i in range(df.shape[0])]
        # Compute geodesic distance between cardholder and merchant
        if parallel:
            # Run in parallel
            with ThreadPoolExecutor() as executor:
                dist = list(executor.map(compute_distance, range(df.shape[0])))
        else:
            dist = [geopy.distance.geodesic(user[i], merch[i]).km for i in range(df.shape[0])]
        del lat, long, merch_lat, merch_long, user, merch
        df['dist'] = dist
        # Sort distances in ascending order
        dist = np.asarray(dist)
        dist.sort()
        # Split array of distances into equal sized groups
        grp = 10
        dist = np.array_split(dist, grp)
        # Dictionary keys correspond to the average within each interval
        distDict = dict.fromkeys([np.mean(_) for _ in dist])
        for dist_ in dist:
            # Extract rows of dataframe which lie between maximum and minimum distance within respective interval
            a = df[df['dist'].between(dist_[0], dist_[-1])]
            # Get the number of fraudulent and non-fraudulent cases
            a = a['is_fraud'].value_counts()
            # Assign value to respective key in dictionary
            distDict[np.mean(dist_)] = fraudShare(a)
        del a, dist_, dist, grp
        # Plot bar chart of fraudulent cases by distance interval
        x, y = zip(*sorted(distDict.items()))
        plt.figure(figsize=(1500 / dpi, 750 / dpi), dpi=dpi)
        plt.bar(x, y)
        plt.ylabel(ylabel="Percentage")
        plt.grid(True, which="both", ls=":")
        plt.xticks()
        plt.title('Percentage of Fraudulent Cases by Distance')
        plt.savefig('dist.png', bbox_inches="tight", dpi=dpi)
        plt.show()


    def plotCatAmtData(df):
        # Scatter plot of amount spent for each category, and whether it is fraudulent
        plt.figure(figsize=(1500 / dpi, 750 / dpi), dpi=dpi)
        df.plot.scatter(x='category', y='amt', c='is_fraud', cmap='viridis')
        plt.ylabel(ylabel="Amount")
        plt.xticks(rotation=90)
        plt.title('Scatter Plot of Amount Spent per Category')
        plt.savefig('catAmt.png', bbox_inches="tight", dpi=dpi)
        plt.show()


    def knnFun():
        # TODO Implement the k nearest neighbour algorithm
        pass


    # -------------------------------------------Start of preamble -----------------------------------------------------
    rng = np.random.default_rng(123542871981236)
    dpi = 96

    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    # Shuffle the datasets while loading it
    # TODO Normalise datasets
    _ = pd.read_csv(os.path.join(THIS_FOLDER, 'data/fraudTrain.csv')).sample(frac=1, random_state=42)
    # Split data into training and validation set
    trainSet, valSet = _[:int(0.9 * _.shape[0])], _[int(0.9 * _.shape[0]):]
    testSet = pd.read_csv(os.path.join(THIS_FOLDER, 'data/fraudTest.csv')).sample(frac=1, random_state=42)
    # Concatenated training, validation and test set for analysis
    conSet = pd.concat([trainSet, valSet, testSet], axis=0)
    # ----------------------------------------------End of preamble ----------------------------------------------------

    # Choose whether to use all or a truncated version of the data set
    simplify = True
    if simplify:
        trainSet = trainSet[:int(9e2)]
        valSet = valSet[:int(1e2)]
        testSet = testSet[:int(1e2)]
        conSet = conSet[:int(1e4)]

    # Decide if parallel computing should be used
    parallel = False
    graphName = 'graph'
    plotFromCsv = False

    # TODO Use clustering algorithm to detect fraud
    if not plotFromCsv:
        tic = time.time()

        # plotCatData(conSet)
        # plotAgeData(conSet)
        # plotStateData(conSet)
        # plotJobData(conSet)
        # plotDistData(conSet, parallel)
        plotCatAmtData(conSet)
        # knnFun()

        toc = time.time()
        print("All simulations completed. Program terminating. Total time taken was",
              str(datetime.timedelta(seconds=toc - tic)))
    else:
        pass
    pass
