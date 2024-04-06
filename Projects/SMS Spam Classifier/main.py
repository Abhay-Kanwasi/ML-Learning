import numpy as np
import pandas as pd

df = pd.read_csv('spam.csv', encoding='ISO-8859-1')  # encoding added to read the csv file

print(df.sample(5))
