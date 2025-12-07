import pandas as pd
import xarray as xr
import numpy as np
from load_climate_data import load_config, load_years
import sys

def check_alignment():
    print("Checking alignment logic...")
    config = load_config("configs/config_climate_wind.yaml")
    
    # Load just one year to be fast
    year = config["data"]["train_years"][0]
    print(f"Loading year {year}...")
    
    # We want to inspect the internals, but load_years hides them.
    # However, load_years returns X, y, meta.
    # If alignment is correct, 'meta' (from target) should match the space-time structure 
    # that the features *should* have.
    
    # But to really check if features match targets, we need to look closer.
    # Let's rely on adding debug prints/checks inside load_climate_data.py 
    # OR running a modified flow here.
    
    # Let's run load_years and assume if I add checks in load_climate_data it will trigger.
    # But I haven't added them yet. I will add them in the next step.
    
    # For now, let's just run it and see if it crashes or produces weird stats.
    X, y, meta, _ = load_years([year], config, return_hard_labels=True, return_metadata=True, align_to_targets=True, verbose=True)
    
    print(f"Loaded {len(X)} rows.")
    print("Sample meta:")
    print(meta.head())
    
    # We can't easily verify feature correctness against target without ground truth correlation.
    # But we can verify if the merge logic had issues if we insert checks in the library.
    
if __name__ == "__main__":
    check_alignment()
