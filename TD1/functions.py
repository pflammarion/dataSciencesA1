import pandas as pd

def standardisation(inputVal):
    average = inputVal.mean()
    standardDeviation = inputVal.std()
    return (inputVal-average)/standardDeviation