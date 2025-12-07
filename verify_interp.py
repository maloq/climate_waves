
import sys
import pandas as pd
from load_climate_data import load_config, load_years

def verify():
    print("Loading config...")
    config = load_config("config_climate.yaml")
    
    # Restrict to 1 year and small region to speed up test
    config["data"]["train_years"] = [2017]
    # config["data"]["latitude_range"] = [40.0, 42.0] # Small slice
    # config["data"]["longitude_range"] = [30.0, 32.0] # Small slice
    
    # We expect roughly:
    # 365 days * (2/0.25 +1 = 9) * (2/0.25 +1 = 9) = 365 * 81 = 29565 rows
    # Actually let's just run it as is, but maybe 1 year is enough?
    # The user said "4 million rows" for presumably 6 years (2017-2022).
    # So 1 year should be ~660k rows. 
    # With original code it was ~4M total / 6 = 660k per year.
    # WAIT. User said "4 million rows" which was LOW.
    # Expected was 64 million. So ~10M per year.
    # With my fix, for 1 year, we expect ~10M rows (full grid).
    # To avoid OOM or slow run, let's just use 1 month if possible, but load_years takes list of years.
    # So I'll use 1 year but maybe constrain lat/lon just a bit to verify density change?
    # Or just run for 1 year and see if we get ~10M rows instead of ~600k.
    
    print("Loading 2017 data with interpolation...")
    X, y, y_hard = load_years([2017], config, return_hard_labels=True, verbose=True)
    
    print(f"Loaded DataFrame shape: {X.shape}")
    print(f"Rows: {len(X)}")
    
    # Check density
    # Expected: (70-40)/0.25 + 1 = 121 lats
    # (180-30)/0.25 + 1 = 601 lons
    # 121 * 601 = 72721 points per day
    # * 365 = 26,543,165 rows for 2017.
    
    expected_rows = 121 * 601 * 365
    print(f"Expected rows (approx): {expected_rows}")
    
    if len(X) > 20_000_000:
        print("SUCCESS: Row count is in the expected range (> 20M).")
    elif len(X) > 1_000_000:
        print(f"WARNING: Row count {len(X)} is higher than before but lower than expected full grid.")
    else:
        print("FAILURE: Row count seems too low (similar to before).")

if __name__ == "__main__":
    verify()
