import pandas as pd


def standardisation(input_val):
    average = input_val.mean()
    standard_deviation = input_val.std()
    return (input_val - average) / standard_deviation


# Load the dataset
df = pd.read_table("exercice.csv", delimiter=";")

stand = df
stand['X1'] = standardisation(df['X1'])

stand['X2'] = standardisation(df['X2'])

print(df)
