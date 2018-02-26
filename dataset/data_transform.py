import pandas as pd


df = pd.read_csv('btc.csv')
df["serial_number"] = df.index
df.to_csv('btc_indexed.csv')