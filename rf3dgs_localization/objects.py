from __future__ import absolute_import

import math

import numpy as np

from .contracts import ObjectKind, ObjectTemplate


def _axis_samples(half_extent, spacing):
    count = max(2, int(math.ceil((2.0 * half_extent) / spacing)) + 1)
    return np.linspace(-half_extent, half_extent, count, dtype=np.float64)


def build_cuboid_template(
    kind,
    dimensions_m,
    center_height_m,
    tag_ids,
    tag_offsets_m,
    surface_spacing_m=0.08,
):
    """将规则箱体表面离散为各向异性动态Gaussian。"""
    dimensions = np.asarray(dimensions_m, dtype=np.float64)
    if dimensions.shape != (3,) or np.any(dimensions <= 0.0):
        raise ValueError("箱体尺寸必须是三个正数")
    if surface_spacing_m <= 0.0:
        raise ValueError("surface_spacing_m必须大于0")
    half = dimensions / 2.0
    xyz_blocks = []
    normal_blocks = []
    covariance_blocks = []
    normal_scale = min(0.02, surface_spacing_m * 0.25)
    tangent_scale = surface_spacing_m * 0.55

    for normal_axis in range(3):
        tangent_axes = [axis for axis in range(3) if axis != normal_axis]
        first = _axis_samples(half[tangent_axes[0]], surface_spacing_m)
        second = _axis_samples(half[tangent_axes[1]], surface_spacing_m)
        aa, bb = np.meshgrid(first, second, indexing="ij")
        for sign in (-1.0, 1.0):
            points = np.zeros((aa.size, 3), dtype=np.float64)
            points[:, normal_axis] = sign * half[normal_axis]
            points[:, tangent_axes[0]] = aa.ravel()
            points[:, tangent_axes[1]] = bb.ravel()
            normals = np.zeros_like(points)
            normals[:, normal_axis] = sign
            scales = np.full((len(points), 3), tangent_scale, dtype=np.float64)
            scales[:, normal_axis] = normal_scale
            covariance = np.zeros((len(points), 3, 3), dtype=np.float64)
            covariance[:, np.arange(3), np.arange(3)] = scales * scales
            xyz_blocks.append(points)
            normal_blocks.append(normals)
            covariance_blocks.append(covariance)

    xyz = np.concatenate(xyz_blocks, axis=0)
    normals = np.concatenate(normal_blocks, axis=0)
    covariance = np.concatenate(covariance_blocks, axis=0)
    # [相对介电常数, 等效电导率, 镜面反射先验, 漫反射先验]
    material_prior = (
        np.array([2.5, 0.01, 0.15, 0.45], dtype=np.float64)
        if kind == ObjectKind.PAPER_BOX.value
        else np.array([50.0, 1.0e6, 0.90, 0.10], dtype=np.float64)
    )
    return ObjectTemplate(
        kind=str(kind),
        dimensions_m=dimensions,
        center_height_m=float(center_height_m),
        xyz_m=xyz,
        covariance_m2=covariance,
        opacity=np.full(len(xyz), 0.65, dtype=np.float64),
        normal=normals,
        tag_ids=np.asarray(tag_ids, dtype="U32"),
        tag_offsets_m=np.asarray(tag_offsets_m, dtype=np.float64),
        material_prior=material_prior,
    )


def build_default_object_templates(surface_spacing_m=0.08):
    paper = build_cuboid_template(
        ObjectKind.PAPER_BOX.value,
        [0.5, 0.5, 0.5],
        center_height_m=0.25,
        tag_ids=["53000239", "5300050A", "AAAA0013", "AAAA0014"],
        # README/CSV在文件坐标下为Y±0.15；转换后Map-Y符号相反。
        tag_offsets_m=[
            [0.0, 0.15, 0.25],
            [0.0, -0.15, 0.25],
            [-0.15, 0.0, 0.25],
            [0.15, 0.0, 0.25],
        ],
        surface_spacing_m=surface_spacing_m,
    )
    metal = build_cuboid_template(
        ObjectKind.METAL_BOX.value,
        [0.5, 1.5, 1.0],
        center_height_m=0.5,
        tag_ids=["AAAA0013", "AAAA0014"],
        tag_offsets_m=[[-0.15, 0.0, 0.0], [0.15, 0.0, 0.0]],
        surface_spacing_m=surface_spacing_m,
    )
    return {paper.kind: paper, metal.kind: metal}


def rotate_template(template, center_xyz_m, yaw_rad):
    center = np.asarray(center_xyz_m, dtype=np.float64)
    cosine = math.cos(float(yaw_rad))
    sine = math.sin(float(yaw_rad))
    rotation = np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])
    xyz = template.xyz_m @ rotation.T + center
    covariance = np.einsum(
        "ij,njk,lk->nil", rotation, template.covariance_m2, rotation
    )
    normals = template.normal @ rotation.T
    offsets = template.tag_offsets_m @ rotation.T
    return xyz, covariance, normals, offsets

