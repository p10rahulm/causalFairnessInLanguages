import time
import pandas as pd



if __name__ == "__main__":
    startTime = time.time()

    original = pd.read_csv("datasets/origDev.csv",delimiter="\t", header=0)
    modified = pd.read_csv("datasets/modDev.csv",delimiter="\t", header=0)

    origNeg = original




    timeTaken = time.time() - startTime
    print("Time Taken:", timeTaken)