import pandas as pd


def standardisation(inputVal):
    average = inputVal.mean()
    standardDeviation = inputVal.std()
    return (inputVal - average) / standardDeviation


# Load the dataset
df = pd.read_table("exercice.csv", delimiter=";")

stand = df
stand['X1'] = standardisation(df['X1'])

stand['X2'] = standardisation(df['X2'])

print(df)
