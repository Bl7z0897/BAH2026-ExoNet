import numpy as np

data = np.load(r"E:\ISRO Exoplanet Hackthon\Phase One\Sara\sara\kepler_processed_dataset.npz")
labels = data['labels']
print(f"Total: {len(labels)}")
print(f"Planets (1): {(labels == 1).sum()}")
print(f"Non-planets (0): {(labels == 0).sum()}")