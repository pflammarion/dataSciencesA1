import math
import pandas as pd

def Standardisation(InputVal):
    Average = pd.mean(InputVal)
    StandardDeviation = pd.std(InputVal)
    return (InputVal-Average)/StandardDeviation