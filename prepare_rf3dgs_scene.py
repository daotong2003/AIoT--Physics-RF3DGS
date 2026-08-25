from __future__ import absolute_import

import argparse
import hashlib
import json
from pathlib import Path

from rf3dgs_localization.contracts import RadioGeometry
from rf3dgs_localization.dataset import prepare_measurement_dataset
from rf3dgs_localization.objects import build_default_object_templates
from rf3dgs_localization.scene import prepare_static_scene


PROJECT_ROOT = Path(__file__).resolve().parent


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="构建AIoT静态/动态RF-3DGS场景")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT.parent / "难题数据2",
        help="原始测量和设备坐标目录",
    )
    parser.add_argument(
        "--scene-ply",
        type=Path,
        default=None,
        help="环境点云；未指定时使用数据目录内AIOT_scene.ply",
    )
    parser.add_argument("--output-root", type=Path, required=True, help="云端processed目录")
    parser.add_argument("--max-gaussians", type=int, default=200000, help="静态Gaussian数量上限")
    parser.add_argument("--fine-voxel-m", type=float, default=0.10, help="定位区域体素尺寸，单位m")
    parser.add_argument("--coarse-voxel-m", type=float, default=0.20, help="场景外围体素尺寸，单位m")
    parser.add_argument("--object-spacing-m", type=float, default=0.08, help="动态箱体表面Gaussian间距")
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    scene_ply = args.scene_ply or (
        data_root / "04场景点云" / "成都餐厅点云" / "8_21_2026" / "AIOT_scene.ply"
    )
    observations, data_manifest = prepare_measurement_dataset(
        data_root,
        output / "observation_units.csv",
        output / "measurement_manifest.json",
    )
    xmin = float(observations["object_center_x_map"].min() - 1.0)
    xmax = float(observations["object_center_x_map"].max() + 1.0)
    ymin = float(observations["object_center_y_map"].min() - 1.0)
    ymax = float(observations["object_center_y_map"].max() + 1.0)
    bounds = [xmin, xmax, ymin, ymax]
    scene = prepare_static_scene(
        scene_ply,
        output / "static_scene_gaussians.npz",
        roi_xy_bounds=bounds,
        max_gaussians=args.max_gaussians,
        fine_voxel_m=args.fine_voxel_m,
        coarse_voxel_m=args.coarse_voxel_m,
    )
    templates = build_default_object_templates(args.object_spacing_m)
    template_dir = output / "object_templates"
    for kind, template in templates.items():
        template.save(template_dir / (kind + ".npz"))
    geometry = RadioGeometry.from_csv(data_root / "03prru_cw_position.csv")
    geometry.save(output / "radio_geometry.npz")
    manifest = {
        "schema_version": "1.0",
        "scene_ply": str(scene_ply.resolve()),
        "scene_sha256": _file_hash(scene_ply),
        "radio_geometry_sha256": _file_hash(data_root / "03prru_cw_position.csv"),
        "readme_sha256": _file_hash(data_root / "README.docx"),
        "coordinate_policy": "x_map=x_file, y_map=-y_file",
        "tag_height_m": 0.5,
        "static_gaussian_count": int(len(scene.xyz_m)),
        "max_gaussians": int(args.max_gaussians),
        "fine_voxel_m": float(args.fine_voxel_m),
        "coarse_voxel_m": float(args.coarse_voxel_m),
        "object_spacing_m": float(args.object_spacing_m),
        "localization_bounds_xy_m": bounds,
        "radio_geometry": geometry.to_dict(),
        "measurement_manifest": data_manifest,
    }
    (output / "scene_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
