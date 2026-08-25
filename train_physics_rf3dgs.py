from __future__ import absolute_import

import argparse
import json
from pathlib import Path

import pandas as pd

from rf3dgs_localization.contracts import GaussianSet, ObjectKind, ObjectTemplate, RadioGeometry
from rf3dgs_localization.feature_grid import StaticFeatureGrid, build_static_feature_grid
from rf3dgs_localization.physics import require_torch, torch
from rf3dgs_localization.training import server_preflight, train_physical_model


def main():
    parser = argparse.ArgumentParser(description="训练物理约束静态—动态RF-3DGS")
    parser.add_argument("--processed-root", type=Path, required=True, help="场景预处理目录")
    parser.add_argument("--output-root", type=Path, required=True, help="训练运行目录")
    parser.add_argument("--grid-step-m", type=float, default=0.25, help="离线物理特征网格间距")
    parser.add_argument("--batch-size", type=int, default=64, help="单CW训练单位批量")
    parser.add_argument("--patience-steps", type=int, default=3000, help="各阶段早停等待步数")
    parser.add_argument("--seed", type=int, default=19, help="随机种子")
    parser.add_argument(
        "--stage-steps-json",
        default=None,
        help='覆盖阶段步数，例如 {"link":100,"material":100,"rf":100,"object":100,"joint":100}',
    )
    parser.add_argument("--smoke-test", action="store_true", help="每阶段仅运行2步")
    args = parser.parse_args()

    require_torch()
    report = server_preflight(
        require_cuda=not args.smoke_test,
        require_rasterizer=not args.smoke_test,
        minimum_memory_gb=20.0 if not args.smoke_test else 0.0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processed = args.processed_root.resolve()
    output = args.output_root.resolve()
    manifest = json.loads((processed / "scene_manifest.json").read_text(encoding="utf-8"))
    scene = GaussianSet.load(processed / "static_scene_gaussians.npz")
    geometry = RadioGeometry.load(processed / "radio_geometry.npz")
    templates = {
        kind: ObjectTemplate.load(processed / "object_templates" / (kind + ".npz"))
        for kind in (ObjectKind.PAPER_BOX.value, ObjectKind.METAL_BOX.value)
    }
    observations = pd.read_csv(processed / "observation_units.csv", dtype={"tag_id": str})
    grid_path = processed / "static_rf_feature_grid.npz"
    if grid_path.exists():
        feature_grid = StaticFeatureGrid.load(grid_path)
    else:
        feature_grid = build_static_feature_grid(
            scene,
            geometry,
            manifest["localization_bounds_xy_m"],
            args.grid_step_m,
            tag_height_m=0.5,
            device=device,
            point_batch=64,
            segment_chunk=64,
            gaussian_chunk=32768,
        )
        feature_grid.save(grid_path)
    stage_steps = json.loads(args.stage_steps_json) if args.stage_steps_json else None
    if args.smoke_test:
        stage_steps = {stage: 2 for stage in ("link", "material", "rf", "object", "joint")}
    _, training_manifest = train_physical_model(
        observations,
        feature_grid,
        geometry,
        templates,
        output,
        device,
        stage_steps=stage_steps,
        batch_size=args.batch_size,
        patience_steps=args.patience_steps,
        seed=args.seed,
        use_amp=True,
        provenance={
            "scene_sha256": manifest.get("scene_sha256"),
            "measurement_sha256": manifest.get("measurement_manifest", {}).get("data_sha256"),
            "radio_geometry_sha256": manifest.get("radio_geometry_sha256"),
            "readme_sha256": manifest.get("readme_sha256"),
        },
    )
    summary = {"preflight": report, "training": training_manifest}
    (output / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
