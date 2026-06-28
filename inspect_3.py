import numpy as np

for fname in [
    r"E:\ISRO Exoplanet Hackthon\Phase One\Sara\sara\kepler_dr25_confirmed_2000_batch.npz",
    r"E:\ISRO Exoplanet Hackthon\Phase One\Sara\sara\kepler_dr25_synthetic_2000_batch.npz"
]:
    data = np.load(fname)
    print(f"\n{fname.split(chr(92))[-1]}")
    print(f"  Keys:   {list(data.keys())}")
    print(f"  Flux shape: {data['flux'].shape}")
    print(f"  Labels: {data['labels'].sum():.0f} planets / {len(data['labels'])} total")
    print(f"  Range:  {data['flux'].min():.4f} to {data['flux'].max():.4f}")