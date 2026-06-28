import numpy as np

data = np.load(r"E:\ISRO Exoplanet Hackthon\Phase One\Sara\sara\kepler_processed_dataset.npz")
print(list(data.keys()))
print(data['flux'].shape)
print(data['flux'].dtype)
print(data['labels'].shape)
print(data['flux'].min(), data['flux'].max())