from __future__ import absolute_import

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd


PathLike = Union[str, Path]
MAP_BASIS_FROM_FILE = np.diag([1.0, -1.0, 1.0]).astype(np.float64)
TAG_HEIGHT_M = 0.5


class ObjectKind(str, Enum):
    PAPER_BOX = "paper_box"
    METAL_BOX = "metal_box"


def _coordinate_array(values):
    array = np.asarray(values, dtype=np.float64)
    if array.shape[-1:] != (3,):
        raise ValueError("坐标最后一维必须为3")
    return array


def file_to_map_coordinates(values):
    """将位置文件的向下Y正方向转换到点云全局坐标。"""
    return _coordinate_array(values) @ MAP_BASIS_FROM_FILE.T


def map_to_file_coordinates(values):
    """坐标反射矩阵自逆，因此逆变换与正变换相同。"""
    return _coordinate_array(values) @ MAP_BASIS_FROM_FILE.T


@dataclass
class GaussianSet:
    xyz_m: np.ndarray
    covariance_m2: np.ndarray
    opacity: np.ndarray
    normal: np.ndarray
    geometry_confidence: np.ndarray
    color_rgb: np.ndarray
    material_id: np.ndarray

    def __post_init__(self):
        count = len(self.xyz_m)
        expected = {
            "xyz_m": (count, 3),
            "covariance_m2": (count, 3, 3),
            "opacity": (count,),
            "normal": (count, 3),
            "geometry_confidence": (count,),
            "color_rgb": (count, 3),
            "material_id": (count,),
        }
        for name, shape in expected.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape:
                raise ValueError("%s形状应为%s，实际为%s" % (name, shape, value.shape))
        if count and not np.isfinite(self.xyz_m).all():
            raise ValueError("Gaussian中心包含非有限值")
        if count:
            eigenvalues = np.linalg.eigvalsh(self.covariance_m2)
            if not np.isfinite(eigenvalues).all() or np.any(eigenvalues <= 0.0):
                raise ValueError("Gaussian协方差必须正定")

    def save(self, path):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            xyz_m=np.asarray(self.xyz_m, dtype=np.float32),
            covariance_m2=np.asarray(self.covariance_m2, dtype=np.float32),
            opacity=np.asarray(self.opacity, dtype=np.float32),
            normal=np.asarray(self.normal, dtype=np.float32),
            geometry_confidence=np.asarray(self.geometry_confidence, dtype=np.float32),
            color_rgb=np.asarray(self.color_rgb, dtype=np.uint8),
            material_id=np.asarray(self.material_id, dtype=np.int16),
        )

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            return cls(**{name: archive[name] for name in archive.files})


@dataclass
class ObjectTemplate:
    kind: str
    dimensions_m: np.ndarray
    center_height_m: float
    xyz_m: np.ndarray
    covariance_m2: np.ndarray
    opacity: np.ndarray
    normal: np.ndarray
    tag_ids: np.ndarray
    tag_offsets_m: np.ndarray
    material_prior: np.ndarray

    def save(self, path):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            kind=np.asarray(self.kind),
            dimensions_m=np.asarray(self.dimensions_m, dtype=np.float32),
            center_height_m=np.asarray(self.center_height_m, dtype=np.float32),
            xyz_m=np.asarray(self.xyz_m, dtype=np.float32),
            covariance_m2=np.asarray(self.covariance_m2, dtype=np.float32),
            opacity=np.asarray(self.opacity, dtype=np.float32),
            normal=np.asarray(self.normal, dtype=np.float32),
            tag_ids=np.asarray(self.tag_ids, dtype="U32"),
            tag_offsets_m=np.asarray(self.tag_offsets_m, dtype=np.float32),
            material_prior=np.asarray(self.material_prior, dtype=np.float32),
        )

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            return cls(
                kind=str(archive["kind"].item()),
                dimensions_m=archive["dimensions_m"],
                center_height_m=float(archive["center_height_m"].item()),
                xyz_m=archive["xyz_m"],
                covariance_m2=archive["covariance_m2"],
                opacity=archive["opacity"],
                normal=archive["normal"],
                tag_ids=archive["tag_ids"],
                tag_offsets_m=archive["tag_offsets_m"],
                material_prior=archive["material_prior"],
            )


@dataclass
class RadioGeometry:
    cw_xyz_m: np.ndarray
    prru_xyz_m: np.ndarray

    def __post_init__(self):
        self.cw_xyz_m = np.asarray(self.cw_xyz_m, dtype=np.float64)
        self.prru_xyz_m = np.asarray(self.prru_xyz_m, dtype=np.float64)
        if self.cw_xyz_m.shape != (4, 3) or self.prru_xyz_m.shape != (4, 3):
            raise ValueError("要求4个CW和4个pRRU坐标")

    @property
    def rx_xyz_m(self):
        # 缺少阵元基线时，两路RX使用所属pRRU设备中心。
        return np.repeat(self.prru_xyz_m, 2, axis=0)

    @classmethod
    def from_csv(cls, path):
        frame = pd.read_csv(path)
        required = {"type", "id", "x", "y", "z"}
        if not required.issubset(frame.columns):
            raise ValueError("设备坐标文件缺少字段: %s" % sorted(required - set(frame.columns)))
        names = frame["type"].astype(str).str.lower()
        cw = frame[names.str.startswith("cwant")].sort_values("id")
        prru = frame[names.str.startswith("prru")].sort_values("id")
        return cls(
            file_to_map_coordinates(cw[["x", "y", "z"]].to_numpy(float)),
            file_to_map_coordinates(prru[["x", "y", "z"]].to_numpy(float)),
        )

    def to_dict(self):
        return {
            "cw_xyz_m": self.cw_xyz_m.tolist(),
            "prru_xyz_m": self.prru_xyz_m.tolist(),
        }

    def save(self, path):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            cw_xyz_m=self.cw_xyz_m.astype(np.float32),
            prru_xyz_m=self.prru_xyz_m.astype(np.float32),
        )

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            return cls(archive["cw_xyz_m"], archive["prru_xyz_m"])
