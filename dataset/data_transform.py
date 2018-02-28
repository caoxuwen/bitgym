import pandas as pd
import numpy as np

df = pd.read_csv('btc.csv')
df["serial_number"] = df["datetime"] % (60*60*24)/60/15
df["serial_number"] = df['serial_number'].astype(np.int64)
df.to_csv('btc_indexed2.csv')