from packages.util import *

if __name__ == '__main__':
    tic = time.time()

    preprocess_data()

    toc = time.time()
    print("All simulations completed. Program terminating. Total time taken was",
          str(datetime.timedelta(seconds=toc - tic)))