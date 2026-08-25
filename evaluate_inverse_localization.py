from __future__ import absolute_import

import argparse
import json
from pathlib import Path

import pandas as pd

from rf3dgs_localization.contracts import ObjectKind, ObjectTemplate, RadioGeometry
from rf3dgs_localization.feature_grid import StaticFeatureGrid
from rf3dgs_localization.inference import (
    calibrate_temperature,
    evaluate_split,
    threshold_metrics,
)
from rf3dgs_localization.physics import require_torch, torch
from rf3dgs_localization.training import load_physical_model


def _serializable_rows(rows):
    return [{key: value for key, value in row.items() if not key.startswith("_") and key != "truth_xy"} for row in rows]


def main():
    parser = argparse.ArgumentParser(description="RF-3DGS物理反演定位与1/3/5m评估")
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=32, help="连续优化候选数")
    parser.add_argument("--refine-steps", type=int, default=50, help="每个候选L-BFGS最大步数")
    parser.add_argument("--smoke-test", action="store_true", help="使用单候选且不连续优化")
    args = parser.parse_args()

    require_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processed = args.processed_root.resolve()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    model, model_manifest = load_physical_model(args.model_dir, device)
    feature_grid = StaticFeatureGrid.load(processed / "static_rf_feature_grid.npz")
    geometry = RadioGeometry.load(processed / "radio_geometry.npz")
    templates = {
        kind: ObjectTemplate.load(processed / "object_templates" / (kind + ".npz"))
        for kind in (ObjectKind.PAPER_BOX.value, ObjectKind.METAL_BOX.value)
    }
    frame = pd.read_csv(processed / "observation_units.csv", dtype={"tag_id": str})
    training_centers = model_manifest.get("training_centers_xy_m", [])
    validation = evaluate_split(
        frame,
        "val",
        model,
        feature_grid,
        geometry,
        templates,
        device,
        temperature=1.0,
        ablation="full",
        top_k=1,
        refine_steps=0,
        training_centers_xy=training_centers,
        return_surfaces=True,
    )
    temperature = calibrate_temperature(validation)
    top_k = 1 if args.smoke_test else args.top_k
    refine_steps = 0 if args.smoke_test else args.refine_steps
    reports = {}
    for ablation in ("free_space", "static", "static_dynamic", "full"):
        rows = evaluate_split(
            frame,
            "test",
            model,
            feature_grid,
            geometry,
            templates,
            device,
            temperature=temperature,
            ablation=ablation,
            top_k=top_k,
            refine_steps=refine_steps,
            training_centers_xy=training_centers,
        )
        clean_rows = _serializable_rows(rows)
        pd.DataFrame(clean_rows).to_csv(
            output / (ablation + "_test_predictions.csv"), index=False, encoding="utf-8-sig"
        )
        reports[ablation] = threshold_metrics(clean_rows)
    full = reports["full"]
    status = "pass" if (
        full.get("success_at_1m", 0.0) >= 0.125
        and full.get("success_at_3m", 0.0) >= 0.65
        and full.get("success_at_5m", 0.0) >= 0.90
    ) else "fail"
    metrics = {
        "status": status,
        "official_thresholds_m": [1.0, 3.0, 5.0],
        "position_target": "object_geometric_center",
        "temperature_selected_on_validation": temperature,
        "test_ablation_metrics": reports,
        "test_evaluated_after_temperature_frozen": True,
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

