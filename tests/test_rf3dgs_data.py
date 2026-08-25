import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from rf3dgs_localization.dataset import load_measurement_csv, prepare_observation_units


class RF3DGSDataContractTest(unittest.TestCase):
    def test_rsrp6_and_physical_coordinates_are_corrected(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "measurements.csv"
            row = {
                "center_point": 1,
                "cent_x": 3.0,
                "cent_y": 4.0,
                "cent_z": 0.5,
                "x": 3.15,
                "y": 4.0,
                "z": 0.0,
                "tag_id": "tag-a",
                "cw_ant_id": 0,
            }
            for antenna in range(1, 9):
                row["rsrp_%d" % antenna] = -6400 if antenna == 6 else -100
                row["sinr_%d" % antenna] = 5
            pd.DataFrame([row, row]).to_csv(path, index=False)

            frame, audit = load_measurement_csv(path)
            self.assertTrue(audit["rsrp_6_divided_by_64"])
            self.assertAlmostEqual(float(frame["rsrp_6"].iloc[0]), -100.0)

            units = prepare_observation_units(frame, object_kind="paper_box")
            self.assertEqual(len(units), 1)
            self.assertAlmostEqual(float(units["tag_z_map"].iloc[0]), 0.5)
            self.assertAlmostEqual(float(units["object_center_y_map"].iloc[0]), -4.0)
            self.assertEqual(units["split"].iloc[0], "test")
            self.assertEqual(int(units["repeat_count"].iloc[0]), 2)
            self.assertTrue(np.isfinite(units.filter(regex="^(rsrp|sinr)_").to_numpy()).all())


if __name__ == "__main__":
    unittest.main()
