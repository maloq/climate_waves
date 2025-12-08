
import unittest
import numpy as np
import xarray as xr
from feature_engineering import compute_regional_features

class TestRegionalFeatures(unittest.TestCase):
    def test_compute_regional_features(self):
        # Create a dummy dataset
        coords = {
            "time": np.arange(10),
            "latitude": np.arange(5),
            "longitude": np.arange(5)
        }
        data = np.random.randn(10, 5, 5)
        # Add a pattern:
        # Time 0: all 0
        data[0, :, :] = 0
        # Time 1: all 1
        data[1, :, :] = 1
        
        ds = xr.Dataset(
            {"temp": (("time", "latitude", "longitude"), data)},
            coords=coords
        )
        
        # Test basic computation
        res = compute_regional_features(ds, ["temp"], stats=["mean", "max", "min"])
        
        # Check global mean for time 0 (should be 0)
        np.testing.assert_allclose(res["temp_global_mean"].isel(time=0).values, 0)
        # Check global mean for time 1 (should be 1)
        np.testing.assert_allclose(res["temp_global_mean"].isel(time=1).values, 1)
        
        # Check shape preservation (broadcasting)
        self.assertEqual(res["temp_global_mean"].shape, ds["temp"].shape)
        
        # Check if values are broadcasted (all pixels in a time slice should have same global meean)
        t1_slice = res["temp_global_mean"].isel(time=5).values
        self.assertTrue(np.all(t1_slice == t1_slice[0,0]))

if __name__ == '__main__':
    unittest.main()
