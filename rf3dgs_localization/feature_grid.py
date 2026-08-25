from __future__ import absolute_import

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .physics import PATH_FEATURE_COUNT, extract_static_path_features, require_torch, torch


@dataclass
class StaticFeatureGrid:
    x_values_m: np.ndarray
    y_values_m: np.ndarray
    tag_height_m: float
    features: np.ndarray

    def __post_init__(self):
        expected = (
            len(self.y_values_m),
            len(self.x_values_m),
            4,
            8,
            PATH_FEATURE_COUNT,
        )
        if self.features.shape != expected:
            raise ValueError("静态特征网格形状应为%s，实际为%s" % (expected, self.features.shape))

    @property
    def step_x_m(self):
        return float(self.x_values_m[1] - self.x_values_m[0])

    @property
    def step_y_m(self):
        return float(self.y_values_m[1] - self.y_values_m[0])

    def save(self, path):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            x_values_m=self.x_values_m.astype(np.float32),
            y_values_m=self.y_values_m.astype(np.float32),
            tag_height_m=np.asarray(self.tag_height_m, dtype=np.float32),
            features=self.features.astype(np.float32),
        )

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            return cls(
                archive["x_values_m"],
                archive["y_values_m"],
                float(archive["tag_height_m"].item()),
                archive["features"],
            )

    def query_torch(self, xy_m, cw_ids, device):
        """双线性查询保持对xy的梯度，越界位置钳制到边界。"""
        require_torch()
        xy = xy_m.to(device=device, dtype=torch.float32)
        ids = cw_ids.to(device=device, dtype=torch.long)
        grid = torch.as_tensor(self.features, dtype=torch.float32, device=device)
        x0 = float(self.x_values_m[0])
        y0 = float(self.y_values_m[0])
        ux = (xy[:, 0] - x0) / self.step_x_m
        uy = (xy[:, 1] - y0) / self.step_y_m
        ix0 = torch.floor(ux).long().clamp(0, len(self.x_values_m) - 2)
        iy0 = torch.floor(uy).long().clamp(0, len(self.y_values_m) - 2)
        ix1 = ix0 + 1
        iy1 = iy0 + 1
        wx = (ux - ix0.float()).clamp(0.0, 1.0)[:, None, None]
        wy = (uy - iy0.float()).clamp(0.0, 1.0)[:, None, None]
        f00 = grid[iy0, ix0, ids]
        f10 = grid[iy0, ix1, ids]
        f01 = grid[iy1, ix0, ids]
        f11 = grid[iy1, ix1, ids]
        return (
            f00 * (1.0 - wx) * (1.0 - wy)
            + f10 * wx * (1.0 - wy)
            + f01 * (1.0 - wx) * wy
            + f11 * wx * wy
        )


def build_static_feature_grid(
    scene,
    geometry,
    bounds_xy_m,
    step_m,
    tag_height_m,
    device,
    point_batch=64,
    segment_chunk=64,
    gaussian_chunk=32768,
):
    require_torch()
    xmin, xmax, ymin, ymax = [float(value) for value in bounds_xy_m]
    x_values = np.arange(xmin, xmax + step_m * 0.5, step_m, dtype=np.float64)
    y_values = np.arange(ymin, ymax + step_m * 0.5, step_m, dtype=np.float64)
    xx, yy = np.meshgrid(x_values, y_values, indexing="xy")
    points = np.column_stack(
        [xx.ravel(), yy.ravel(), np.full(xx.size, float(tag_height_m))]
    )
    result = np.empty((len(points), 4, 8, PATH_FEATURE_COUNT), dtype=np.float32)
    for start in range(0, len(points), int(point_batch)):
        selected = points[start : start + int(point_batch)]
        repeated_points = np.repeat(selected, 4, axis=0)
        cw_ids = np.tile(np.arange(4, dtype=np.int64), len(selected))
        with torch.no_grad():
            features = extract_static_path_features(
                repeated_points,
                cw_ids,
                geometry,
                scene,
                device,
                segment_chunk=segment_chunk,
                gaussian_chunk=gaussian_chunk,
            )
        result[start : start + len(selected)] = (
            features.reshape(len(selected), 4, 8, PATH_FEATURE_COUNT).cpu().numpy()
        )
    result = result.reshape(
        len(y_values), len(x_values), 4, 8, PATH_FEATURE_COUNT
    )
    return StaticFeatureGrid(x_values, y_values, float(tag_height_m), result)
