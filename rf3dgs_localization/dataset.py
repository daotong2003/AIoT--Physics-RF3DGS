from __future__ import absolute_import

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import ObjectKind, TAG_HEIGHT_M


MEASUREMENT_COLUMNS = tuple(
    column
    for antenna in range(1, 9)
    for column in ("rsrp_%d" % antenna, "sinr_%d" % antenna)
)


def load_measurement_csv(path):
    source = Path(path)
    frame = pd.read_csv(source, dtype={"tag_id": str})
    required = {
        "tag_id",
        "cw_ant_id",
        "center_point",
        "cent_x",
        "cent_y",
        "x",
        "y",
    }.union(MEASUREMENT_COLUMNS)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("%s缺少必要列: %s" % (source.name, missing))
    result = frame.copy()
    result["tag_id"] = result["tag_id"].astype(str)
    result["cw_ant_id"] = pd.to_numeric(result["cw_ant_id"], errors="raise").astype(int)
    rsrp6 = pd.to_numeric(result["rsrp_6"], errors="coerce")
    valid = rsrp6[np.isfinite(rsrp6)]
    # 原始LOS/NLOS库的第6路采用定点放大值，数据契约要求读取时始终还原。
    result["rsrp_6"] = rsrp6 / 64.0
    for column in MEASUREMENT_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result, {
        "source": str(source.resolve()),
        "row_count": int(len(result)),
        "rsrp_6_divided_by_64": True,
        "rsrp_6_raw_median": float(valid.median()) if len(valid) else None,
    }


def spatial_split(center_points):
    ids = pd.to_numeric(center_points, errors="raise").astype(int).to_numpy()
    row = (ids - 1) // 6
    column = (ids - 1) % 6
    code = (row + 2 * column) % 5
    split = np.full(len(ids), "train", dtype="U5")
    split[code == 1] = "val"
    split[code == 0] = "test"
    return split


def prepare_observation_units(frame, object_kind):
    if object_kind not in (ObjectKind.PAPER_BOX.value, ObjectKind.METAL_BOX.value):
        raise ValueError("未知物体类型: %s" % object_kind)
    keys = [
        "center_point",
        "cent_x",
        "cent_y",
        "tag_id",
        "x",
        "y",
        "cw_ant_id",
    ]
    grouped = frame.groupby(keys, sort=True, dropna=False)
    units = grouped[list(MEASUREMENT_COLUMNS)].median().reset_index()
    units["repeat_count"] = grouped.size().to_numpy(dtype=int)
    units["object_kind"] = object_kind
    units["object_kind_index"] = 0 if object_kind == ObjectKind.PAPER_BOX.value else 1
    units["object_center_x_map"] = units["cent_x"].astype(float)
    units["object_center_y_map"] = -units["cent_y"].astype(float)
    units["object_center_z_map"] = 0.25 if object_kind == ObjectKind.PAPER_BOX.value else 0.5
    units["tag_x_map"] = units["x"].astype(float)
    units["tag_y_map"] = -units["y"].astype(float)
    units["tag_z_map"] = TAG_HEIGHT_M
    units["tag_offset_x_map"] = units["tag_x_map"] - units["object_center_x_map"]
    units["tag_offset_y_map"] = units["tag_y_map"] - units["object_center_y_map"]
    units["tag_offset_z_map"] = units["tag_z_map"] - units["object_center_z_map"]
    units["split"] = spatial_split(units["center_point"])
    return units


def _sha256(paths):
    digest = hashlib.sha256()
    for path in paths:
        with Path(path).open("rb") as stream:
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    return digest.hexdigest()


def prepare_measurement_dataset(data_root, output_csv, manifest_path):
    root = Path(data_root)
    sources = [
        (ObjectKind.PAPER_BOX.value, root / "01参考标签信息" / "df_train_lib_LOS.csv"),
        (ObjectKind.METAL_BOX.value, root / "01参考标签信息" / "df_train_lib_NLOS.csv"),
    ]
    units = []
    audits = {}
    for object_kind, path in sources:
        frame, audit = load_measurement_csv(path)
        prepared = prepare_observation_units(frame, object_kind)
        units.append(prepared)
        audits[object_kind] = dict(audit, unit_count=int(len(prepared)))
    result = pd.concat(units, ignore_index=True)
    result["parent_group_id"] = pd.factorize(
        pd.MultiIndex.from_frame(
            result[["object_kind", "center_point", "tag_id", "object_center_x_map", "object_center_y_map"]]
        ),
        sort=True,
    )[0]
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, encoding="utf-8-sig")
    manifest = {
        "schema_version": "1.0",
        "coordinate_policy": "x_map=x_file, y_map=-y_file, z_tag=0.5m",
        "input_policy": "one CW + four pRRU (8 RX x RSRP/SINR)",
        "data_sha256": _sha256([path for _, path in sources]),
        "sources": audits,
        "unit_counts": {
            split: int((result["split"] == split).sum()) for split in ("train", "val", "test")
        },
        "parent_group_counts": {
            split: int(result.loc[result["split"] == split, "parent_group_id"].nunique())
            for split in ("train", "val", "test")
        },
    }
    target_manifest = Path(manifest_path)
    target_manifest.parent.mkdir(parents=True, exist_ok=True)
    target_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return result, manifest
