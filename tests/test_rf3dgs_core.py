import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from rf3dgs_localization.contracts import (
    ObjectKind,
    file_to_map_coordinates,
    map_to_file_coordinates,
)
from rf3dgs_localization.objects import build_default_object_templates
from rf3dgs_localization.scene import prepare_static_scene, voxelize_points


class CoordinateContractTest(unittest.TestCase):
    def test_file_map_coordinate_round_trip(self):
        points = np.array([[1.0, 2.0, 0.5], [-3.0, -4.0, 5.2]])
        mapped = file_to_map_coordinates(points)
        np.testing.assert_allclose(mapped, [[1.0, -2.0, 0.5], [-3.0, 4.0, 5.2]])
        np.testing.assert_allclose(map_to_file_coordinates(mapped), points)


class ObjectTemplateTest(unittest.TestCase):
    def test_default_templates_obey_geometry_and_height_contract(self):
        templates = build_default_object_templates(surface_spacing_m=0.25)
        paper = templates[ObjectKind.PAPER_BOX.value]
        metal = templates[ObjectKind.METAL_BOX.value]

        np.testing.assert_allclose(paper.dimensions_m, [0.5, 0.5, 0.5])
        np.testing.assert_allclose(metal.dimensions_m, [0.5, 1.5, 1.0])
        self.assertAlmostEqual(paper.center_height_m, 0.25)
        self.assertAlmostEqual(metal.center_height_m, 0.5)
        self.assertTrue(np.allclose(paper.tag_offsets_m[:, 2], 0.25))
        self.assertTrue(np.allclose(metal.tag_offsets_m[:, 2], 0.0))
        self.assertTrue(np.allclose(
            paper.center_height_m + paper.tag_offsets_m[:, 2], 0.5
        ))
        self.assertTrue(np.allclose(
            metal.center_height_m + metal.tag_offsets_m[:, 2], 0.5
        ))
        self.assertGreater(len(paper.xyz_m), 0)
        self.assertGreater(len(metal.xyz_m), len(paper.xyz_m))


class GaussianSceneTest(unittest.TestCase):
    def test_scene_ply_coordinates_are_not_flipped(self):
        content = """ply
format ascii 1.0
element vertex 3
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
0.00 2.00 0.00 255 0 0
0.04 2.00 0.00 255 0 0
0.00 2.04 0.00 255 0 0
"""
        with TemporaryDirectory() as directory:
            source = Path(directory) / "scene.ply"
            target = Path(directory) / "scene.npz"
            source.write_text(content, encoding="ascii")
            scene = prepare_static_scene(
                source,
                target,
                roi_xy_bounds=(-1.0, 1.0, 1.0, 3.0),
                fine_voxel_m=0.1,
                coarse_voxel_m=0.2,
                add_structural_planes=False,
            )
            self.assertGreater(float(scene.xyz_m[:, 1].mean()), 1.9)

    def test_voxel_gaussian_covariances_are_positive_definite(self):
        points = np.array(
            [
                [0.00, 0.00, 0.00],
                [0.04, 0.00, 0.00],
                [0.00, 0.04, 0.00],
                [1.00, 1.00, 1.00],
            ],
            dtype=np.float64,
        )
        colors = np.full((len(points), 3), 128, dtype=np.uint8)
        gaussians = voxelize_points(points, colors, voxel_size_m=0.1)
        self.assertEqual(gaussians.xyz_m.shape, (2, 3))
        eigenvalues = np.linalg.eigvalsh(gaussians.covariance_m2)
        self.assertTrue(np.isfinite(eigenvalues).all())
        self.assertTrue((eigenvalues > 0.0).all())
        self.assertTrue(((gaussians.opacity > 0.0) & (gaussians.opacity <= 1.0)).all())


if __name__ == "__main__":
    unittest.main()
