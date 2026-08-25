from __future__ import absolute_import

from pathlib import Path

import numpy as np

from .contracts import GaussianSet, file_to_map_coordinates


_PLY_TYPES = {
    "char": "i1",
    "uchar": "u1",
    "int8": "i1",
    "uint8": "u1",
    "short": "i2",
    "ushort": "u2",
    "int16": "i2",
    "uint16": "u2",
    "int": "i4",
    "uint": "u4",
    "int32": "i4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


def load_ply_vertices(path):
    """读取仅含标量vertex属性的ASCII或二进制PLY。"""
    source = Path(path)
    with source.open("rb") as stream:
        header = []
        offset = 0
        while True:
            line = stream.readline()
            if not line:
                raise ValueError("PLY缺少end_header")
            offset += len(line)
            text = line.decode("ascii").strip()
            header.append(text)
            if text == "end_header":
                break
    if not header or header[0] != "ply":
        raise ValueError("不是有效PLY文件")
    format_name = next(line.split()[1] for line in header if line.startswith("format "))
    vertex_count = int(next(line.split()[2] for line in header if line.startswith("element vertex ")))
    properties = []
    in_vertex = False
    for line in header:
        if line.startswith("element vertex "):
            in_vertex = True
            continue
        if in_vertex and line.startswith("element "):
            break
        if in_vertex and line.startswith("property "):
            tokens = line.split()
            if tokens[1] == "list":
                raise ValueError("不支持vertex列表属性")
            properties.append((tokens[2], tokens[1]))
    names = [name for name, _ in properties]
    for required in ("x", "y", "z"):
        if required not in names:
            raise ValueError("PLY缺少%s属性" % required)

    if format_name.startswith("binary"):
        endian = "<" if "little" in format_name else ">"
        dtype = np.dtype(
            [
                (name, ("|" if _PLY_TYPES[type_name] in ("i1", "u1") else endian) + _PLY_TYPES[type_name])
                for name, type_name in properties
            ]
        )
        archive = np.memmap(source, dtype=dtype, mode="r", offset=offset, shape=(vertex_count,))
        xyz = np.column_stack([archive[name] for name in ("x", "y", "z")]).astype(np.float64)
        color_names = ("red", "green", "blue")
        colors = (
            np.column_stack([archive[name] for name in color_names]).astype(np.uint8)
            if set(color_names).issubset(names)
            else np.full((vertex_count, 3), 127, dtype=np.uint8)
        )
    elif format_name == "ascii":
        values = np.loadtxt(source, skiprows=len(header), max_rows=vertex_count)
        lookup = {name: index for index, name in enumerate(names)}
        xyz = values[:, [lookup[name] for name in ("x", "y", "z")]].astype(np.float64)
        colors = (
            values[:, [lookup[name] for name in ("red", "green", "blue")]].astype(np.uint8)
            if {"red", "green", "blue"}.issubset(lookup)
            else np.full((vertex_count, 3), 127, dtype=np.uint8)
        )
    else:
        raise ValueError("不支持PLY格式: %s" % format_name)
    valid = np.isfinite(xyz).all(axis=1)
    return xyz[valid], colors[valid]


def voxelize_points(points_m, colors_rgb, voxel_size_m, geometry_confidence=1.0):
    points = np.asarray(points_m, dtype=np.float64)
    colors = np.asarray(colors_rgb, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or colors.shape != points.shape:
        raise ValueError("points_m和colors_rgb必须同为[N,3]")
    if voxel_size_m <= 0.0 or not len(points):
        if not len(points):
            empty = np.empty((0, 3), dtype=np.float64)
            return GaussianSet(empty, np.empty((0, 3, 3)), np.empty(0), empty, np.empty(0), empty.astype(np.uint8), np.empty(0, dtype=np.int16))
        raise ValueError("voxel_size_m必须大于0")

    keys = np.floor(points / float(voxel_size_m)).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    voxel_count = len(counts)
    sums = np.zeros((voxel_count, 3), dtype=np.float64)
    color_sums = np.zeros((voxel_count, 3), dtype=np.float64)
    np.add.at(sums, inverse, points)
    np.add.at(color_sums, inverse, colors)
    means = sums / counts[:, None]
    color_means = np.clip(color_sums / counts[:, None], 0, 255).astype(np.uint8)

    second = np.zeros((voxel_count, 3, 3), dtype=np.float64)
    for left in range(3):
        for right in range(3):
            np.add.at(second[:, left, right], inverse, points[:, left] * points[:, right])
    covariance = second / counts[:, None, None] - np.einsum("ni,nj->nij", means, means)
    covariance = 0.5 * (covariance + np.transpose(covariance, (0, 2, 1)))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    minimum_variance = max(0.01, voxel_size_m * 0.12) ** 2
    maximum_variance = (voxel_size_m * 0.75) ** 2
    eigenvalues = np.clip(eigenvalues, minimum_variance, maximum_variance)
    covariance = np.einsum("nij,nj,nkj->nik", eigenvectors, eigenvalues, eigenvectors)
    normals = eigenvectors[:, :, 0]
    median_count = max(float(np.median(counts)), 1.0)
    opacity = np.clip(np.sqrt(counts / median_count), 0.1, 1.0)
    return GaussianSet(
        xyz_m=means,
        covariance_m2=covariance,
        opacity=opacity,
        normal=normals,
        geometry_confidence=np.full(voxel_count, float(geometry_confidence)),
        color_rgb=color_means,
        material_id=np.zeros(voxel_count, dtype=np.int16),
    )


def concatenate_gaussians(parts):
    valid = [part for part in parts if len(part.xyz_m)]
    if not valid:
        raise ValueError("没有可合并的Gaussian")
    return GaussianSet(
        xyz_m=np.concatenate([part.xyz_m for part in valid]),
        covariance_m2=np.concatenate([part.covariance_m2 for part in valid]),
        opacity=np.concatenate([part.opacity for part in valid]),
        normal=np.concatenate([part.normal for part in valid]),
        geometry_confidence=np.concatenate([part.geometry_confidence for part in valid]),
        color_rgb=np.concatenate([part.color_rgb for part in valid]),
        material_id=np.concatenate([part.material_id for part in valid]),
    )


def _plane_gaussians(axis, value, ranges, spacing_m, confidence=0.25):
    tangent = [index for index in range(3) if index != axis]
    first = np.arange(ranges[tangent[0]][0], ranges[tangent[0]][1] + spacing_m * 0.5, spacing_m)
    second = np.arange(ranges[tangent[1]][0], ranges[tangent[1]][1] + spacing_m * 0.5, spacing_m)
    aa, bb = np.meshgrid(first, second, indexing="ij")
    xyz = np.zeros((aa.size, 3), dtype=np.float64)
    xyz[:, axis] = float(value)
    xyz[:, tangent[0]] = aa.ravel()
    xyz[:, tangent[1]] = bb.ravel()
    normal = np.zeros_like(xyz)
    normal[:, axis] = 1.0
    scales = np.full((len(xyz), 3), spacing_m * 0.55)
    scales[:, axis] = 0.03
    covariance = np.zeros((len(xyz), 3, 3))
    covariance[:, np.arange(3), np.arange(3)] = scales * scales
    return GaussianSet(
        xyz,
        covariance,
        np.full(len(xyz), 0.15),
        normal,
        np.full(len(xyz), confidence),
        np.full((len(xyz), 3), 96, dtype=np.uint8),
        np.full(len(xyz), -1, dtype=np.int16),
    )


def prepare_static_scene(
    ply_path,
    output_path,
    roi_xy_bounds,
    max_gaussians=200000,
    fine_voxel_m=0.10,
    coarse_voxel_m=0.20,
    add_structural_planes=True,
):
    xyz_file, colors = load_ply_vertices(ply_path)
    xyz = file_to_map_coordinates(xyz_file)
    xmin, xmax, ymin, ymax = [float(value) for value in roi_xy_bounds]
    fine_mask = (
        (xyz[:, 0] >= xmin)
        & (xyz[:, 0] <= xmax)
        & (xyz[:, 1] >= ymin)
        & (xyz[:, 1] <= ymax)
    )
    fine = voxelize_points(xyz[fine_mask], colors[fine_mask], fine_voxel_m)
    coarse = voxelize_points(xyz[~fine_mask], colors[~fine_mask], coarse_voxel_m)
    parts = [fine, coarse]
    if add_structural_planes:
        ranges = {0: (xmin, xmax), 1: (ymin, ymax), 2: (0.0, 5.2)}
        parts.extend(
            [
                _plane_gaussians(2, 0.0, ranges, 0.30),
                _plane_gaussians(2, 5.2, ranges, 0.30),
                _plane_gaussians(1, -20.0, ranges, 0.30),
                _plane_gaussians(1, 2.0, ranges, 0.30),
            ]
        )
    scene = concatenate_gaussians(parts)
    if len(scene.xyz_m) > int(max_gaussians):
        # 优先保留高可信且局部点数更密集的Gaussian，其余确定性抽样。
        score = scene.geometry_confidence * scene.opacity
        order = np.argsort(-score, kind="stable")[: int(max_gaussians)]
        scene = GaussianSet(
            scene.xyz_m[order],
            scene.covariance_m2[order],
            scene.opacity[order],
            scene.normal[order],
            scene.geometry_confidence[order],
            scene.color_rgb[order],
            scene.material_id[order],
        )
    scene.save(output_path)
    return scene
