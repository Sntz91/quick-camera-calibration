import pandas as pd

img = 0
H = 0
validation_points_tv = []
validation_points_pv = []
reference_points_tv = []
reference_points_pv = []

fname = 'DJI_0026'
df = pd.read_pickle(f'out/statistics/{fname}_df.pickle')
print(df)


# MAKE A NOTEBOOK PLZ.
