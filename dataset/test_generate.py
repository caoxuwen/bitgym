import pandas as pd
import numpy as np

df = pd.read_csv('btc_indexed2.csv')
df["low"] = df.index+1000.0
df["high"] = df.index+1000.0
df["open"] = df.index+1000.0
df["close"] = df.index+1000.0
df["volume"] = 100.0
df.to_csv('btc_test.csv')