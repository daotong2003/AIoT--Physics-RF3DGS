from __future__ import absolute_import

import copy
import json
import random
import subprocess
from pathlib import Path

import numpy as np
from .contracts import ObjectKind
from .physics import (
    PhysicalRFModel,
    extract_object_path_features,
    require_torch,
    torch,
)


DEFAULT_STAGE_STEPS = {
    "link": 5000,
    "material": 20000,
    "rf": 40000,
    "object": 20000,
    "joint": 10000,
}

_STAGE_PARAMETERS = {
    "link": {
        "cw_gain_db",
        "rx_gain_db",
        "tag_reflection_db",
        "noise_floor_dbm",
        "log_sigma_rsrp",
        "log_sigma_q",
    },
    "material": {
        "raw_scene_attenuation",
        "raw_reflection_gain",
        "raw_uncertain_scale",
        "raw_q_density",
    },
    "rf": {"residual_basis", "residual_link"},
    "object": {"raw_paper_attenuation", "raw_metal_delta"},
    "joint": None,
}

_STAGE_LEARNING_RATE = {
    "link": 2.0e-2,
    "material": 1.0e-2,
    "rf": 5.0e-3,
    "object": 5.0e-3,
    "joint": 1.0e-3,
}


def server_preflight(require_cuda=True, require_rasterizer=False, minimum_memory_gb=0.0):
    require_torch()
    cuda_available = bool(torch.cuda.is_available())
    if require_cuda and not cuda_available:
        raise RuntimeError("云端训练要求CUDA可用")
    device = torch.device("cuda" if cuda_available else "cpu")
    value = torch.tensor([1.0], device=device, requires_grad=True)
    (value.square().sum()).backward()
    rasterizer_available = False
    try:
        __import__("diff_gaussian_rasterization")
        rasterizer_available = True
    except ImportError:
        pass
    report = {
        "torch_version": str(torch.__version__),
        "cuda_available": cuda_available,
        "cuda_version": str(torch.version.cuda),
        "device": str(device),
        "rasterizer_available": rasterizer_available,
    }
    if cuda_available:
        properties = torch.cuda.get_device_properties(0)
        report.update(
            {
                "gpu_name": properties.name,
                "gpu_memory_bytes": int(properties.total_memory),
            }
        )
        if float(properties.total_memory) < float(minimum_memory_gb) * (1024.0 ** 3):
            raise RuntimeError("GPU显存低于%.1f GB，不能运行正式配置" % float(minimum_memory_gb))
    if require_rasterizer and not rasterizer_available:
        raise RuntimeError("WRF-GS的diff_gaussian_rasterization扩展不可用")
    return report


def _measurement_tensors(frame, device):
    rsrp_columns = ["rsrp_%d" % index for index in range(1, 9)]
    sinr_columns = ["sinr_%d" % index for index in range(1, 9)]
    rsrp = frame[rsrp_columns].to_numpy(dtype=np.float32)
    sinr = frame[sinr_columns].to_numpy(dtype=np.float32)
    q = rsrp - sinr
    valid = np.isfinite(rsrp) & np.isfinite(q)
    return (
        torch.as_tensor(np.where(valid, rsrp, 0.0), device=device),
        torch.as_tensor(np.where(valid, q, 0.0), device=device),
        torch.as_tensor(valid, dtype=torch.bool, device=device),
    )


def build_feature_cache(frame, feature_grid, geometry, templates, device):
    """固定训练位置的物理特征只计算一次，避免每步扫描静态点云。"""
    require_torch()
    cw_ids = torch.as_tensor(frame["cw_ant_id"].to_numpy(int), dtype=torch.long, device=device)
    tag_xy = torch.as_tensor(
        frame[["tag_x_map", "tag_y_map"]].to_numpy(np.float32), device=device
    )
    center_xyz = torch.as_tensor(
        frame[
            ["object_center_x_map", "object_center_y_map", "object_center_z_map"]
        ].to_numpy(np.float32),
        device=device,
    )
    tag_offsets = torch.as_tensor(
        frame[["tag_offset_x_map", "tag_offset_y_map", "tag_offset_z_map"]].to_numpy(
            np.float32
        ),
        device=device,
    )
    object_index = torch.as_tensor(
        frame["object_kind_index"].to_numpy(int), dtype=torch.long, device=device
    )
    with torch.no_grad():
        static_features = feature_grid.query_torch(tag_xy, cw_ids, device)
        object_features = torch.zeros(
            (len(frame), 8, 2), dtype=torch.float32, device=device
        )
        for kind, index in (
            (ObjectKind.PAPER_BOX.value, 0),
            (ObjectKind.METAL_BOX.value, 1),
        ):
            selected = torch.nonzero(object_index == index, as_tuple=False).reshape(-1)
            if len(selected):
                features, _ = extract_object_path_features(
                    center_xyz[selected],
                    torch.zeros(len(selected), device=device),
                    tag_offsets[selected],
                    cw_ids[selected],
                    geometry,
                    templates[kind],
                    device,
                )
                object_features[selected] = features
    observed_rsrp, observed_q, valid_mask = _measurement_tensors(frame, device)
    split_lookup = {"train": 0, "val": 1, "test": 2}
    split = torch.as_tensor(
        [split_lookup[value] for value in frame["split"].astype(str)],
        dtype=torch.long,
        device=device,
    )
    parent = torch.as_tensor(
        frame["parent_group_id"].to_numpy(int), dtype=torch.long, device=device
    )
    return {
        "static_features": static_features,
        "object_features": object_features,
        "object_index": object_index,
        "object_center_xy": center_xyz[:, :2],
        "tag_xy": tag_xy,
        "cw_ids": cw_ids,
        "observed_rsrp": observed_rsrp,
        "observed_q": observed_q,
        "valid_mask": valid_mask,
        "split": split,
        "parent": parent,
    }


def _set_stage_trainable(model, stage):
    selected = _STAGE_PARAMETERS[stage]
    for name, parameter in model.named_parameters():
        parameter.requires_grad = selected is None or name in selected
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def _batch_loss(model, cache, indices, feature_grid, device, stage):
    tag_xy = cache["tag_xy"][indices]
    if stage == "joint":
        tag_xy = tag_xy + model.geometry_offset_xy_m()[None, :]
        static_features = feature_grid.query_torch(
            tag_xy, cache["cw_ids"][indices], device
        )
    else:
        static_features = cache["static_features"][indices]
    prediction = model(
        static_features,
        cache["object_features"][indices],
        cache["object_index"][indices],
        cache["object_center_xy"][indices],
        cache["cw_ids"][indices],
    )
    return model.loss(
        prediction,
        cache["observed_rsrp"][indices],
        cache["observed_q"][indices],
        cache["valid_mask"][indices],
    )


def _evaluate_loss(model, cache, indices, feature_grid, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for start in range(0, len(indices), 256):
            selected = indices[start : start + 256]
            losses.append(
                _batch_loss(model, cache, selected, feature_grid, device, "joint")
                .detach()
                .cpu()
            )
    return float(torch.stack(losses).mean().item()) if losses else float("inf")


def _balanced_probabilities(parent_ids):
    parent = parent_ids.detach().cpu().numpy()
    _, inverse, counts = np.unique(parent, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]
    return weights / weights.sum()


def _git_commit(workdir):
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(workdir),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            universal_newlines=True,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def train_physical_model(
    frame,
    feature_grid,
    geometry,
    templates,
    output_dir,
    device,
    stage_steps=None,
    batch_size=64,
    patience_steps=3000,
    seed=19,
    use_amp=True,
    provenance=None,
):
    require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    steps_by_stage = dict(DEFAULT_STAGE_STEPS)
    if stage_steps:
        steps_by_stage.update({key: int(value) for key, value in stage_steps.items()})
    model = PhysicalRFModel().to(device)
    cache = build_feature_cache(frame, feature_grid, geometry, templates, device)
    train_indices = torch.nonzero(cache["split"] == 0, as_tuple=False).reshape(-1)
    validation_indices = torch.nonzero(cache["split"] == 1, as_tuple=False).reshape(-1)
    if not len(train_indices) or not len(validation_indices):
        raise ValueError("训练集和验证集均不能为空")
    probabilities = _balanced_probabilities(cache["parent"][train_indices])
    rng = np.random.default_rng(seed)
    amp_enabled = bool(use_amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    history = []

    for stage in ("link", "material", "rf", "object", "joint"):
        stage_steps_count = int(steps_by_stage[stage])
        if stage_steps_count <= 0:
            continue
        parameters = _set_stage_trainable(model, stage)
        optimizer = torch.optim.Adam(parameters, lr=_STAGE_LEARNING_RATE[stage])
        validation_interval = max(10, min(200, max(1, stage_steps_count // 10)))
        best_loss = float("inf")
        best_state = copy.deepcopy(model.state_dict())
        best_step = 0
        model.train()
        for step in range(1, stage_steps_count + 1):
            relative = rng.choice(
                len(train_indices),
                size=min(int(batch_size), len(train_indices)),
                replace=True,
                p=probabilities,
            )
            indices = train_indices[
                torch.as_tensor(relative, dtype=torch.long, device=device)
            ]
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                loss = _batch_loss(model, cache, indices, feature_grid, device, stage)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
            if step % validation_interval == 0 or step == stage_steps_count:
                validation_loss = _evaluate_loss(
                    model, cache, validation_indices, feature_grid, device
                )
                history.append(
                    {
                        "stage": stage,
                        "step": step,
                        "train_loss": float(loss.detach().cpu().item()),
                        "validation_loss": validation_loss,
                    }
                )
                if validation_loss < best_loss - 1.0e-5:
                    best_loss = validation_loss
                    best_state = copy.deepcopy(model.state_dict())
                    best_step = step
                if step - best_step >= int(patience_steps):
                    break
                model.train()
        model.load_state_dict(best_state)

    for parameter in model.parameters():
        parameter.requires_grad = False
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_payload = {
        "model_state": model.state_dict(),
        "model_config": {"cw_count": 4, "rx_count": 8, "residual_rank": 2},
    }
    torch.save(model_payload, output / "model.pt")
    training_centers = (
        frame.loc[frame["split"] == "train", ["object_center_x_map", "object_center_y_map"]]
        .drop_duplicates()
        .to_numpy(float)
    )
    manifest = {
        "schema_version": "1.0",
        "model_kind": "physics_constrained_static_dynamic_rf3dgs",
        "position_output": "object_geometric_center",
        "inference_scene_label_policy": "paper/metal hypotheses scored without LOS/NLOS input",
        "tag_height_m": 0.5,
        "coordinate_policy": "x_map=x_file, y_map=-y_file",
        "carrier_frequency_hz": 895000000.0,
        "bandwidth_hz": 180000.0,
        "input_signal_policy": "RSRP/SINR only; incoherent power-domain paths",
        "objective": {
            "rsrp_q_nll": 1.0,
            "material_prior": 1.0e-4,
            "gaussian_rf_smoothness": 1.0e-3,
            "geometry_anchor": 1.0e-2,
        },
        "geometry_center_policy": "frozen feature grid with <=3cm global alignment offset",
        "stage_steps": steps_by_stage,
        "batch_size": int(batch_size),
        "patience_steps": int(patience_steps),
        "mixed_precision": bool(amp_enabled),
        "static_feature_grid_step_m": float(feature_grid.step_x_m),
        "seed": int(seed),
        "temperature": 1.0,
        "training_centers_xy_m": training_centers.tolist(),
        "git_commit": _git_commit(Path.cwd()),
        "runtime": server_preflight(require_cuda=False),
        "history": history,
        "input_provenance": dict(provenance or {}),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return model, manifest


def load_physical_model(model_dir, device):
    require_torch()
    directory = Path(model_dir)
    payload = torch.load(directory / "model.pt", map_location=device)
    model = PhysicalRFModel(**payload["model_config"]).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    return model, manifest
