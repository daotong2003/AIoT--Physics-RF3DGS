from __future__ import absolute_import

import math

import numpy as np

from .contracts import ObjectKind
from .physics import extract_object_path_features, require_torch, torch


def _observations(group, device):
    rsrp = group[["rsrp_%d" % index for index in range(1, 9)]].to_numpy(np.float32)
    sinr = group[["sinr_%d" % index for index in range(1, 9)]].to_numpy(np.float32)
    q = rsrp - sinr
    valid = np.isfinite(rsrp) & np.isfinite(q)
    return (
        torch.as_tensor(np.where(valid, rsrp, 0.0), device=device),
        torch.as_tensor(np.where(valid, q, 0.0), device=device),
        torch.as_tensor(valid, dtype=torch.bool, device=device),
        torch.as_tensor(group["cw_ant_id"].to_numpy(int), dtype=torch.long, device=device),
    )


def _energy_per_candidate(model, prediction, observed_rsrp, observed_q, valid, candidate_count):
    unit_count = len(observed_rsrp)
    rsrp = observed_rsrp[None, :, :].expand(candidate_count, -1, -1)
    q = observed_q[None, :, :].expand(candidate_count, -1, -1)
    mask = valid[None, :, :].expand(candidate_count, -1, -1).float()
    pred_rsrp = prediction["rsrp"].reshape(candidate_count, unit_count, 8)
    pred_q = prediction["q"].reshape(candidate_count, unit_count, 8)
    sigma_rsrp = torch.exp(model.log_sigma_rsrp).clamp(0.5, 20.0)
    sigma_q = torch.exp(model.log_sigma_q).clamp(0.5, 20.0)
    residual_rsrp = (pred_rsrp - rsrp) / sigma_rsrp
    residual_q = (pred_q - q) / sigma_q
    absolute_rsrp = residual_rsrp.abs()
    absolute_q = residual_q.abs()
    huber_rsrp = torch.where(
        absolute_rsrp <= 1.0,
        0.5 * residual_rsrp.square(),
        absolute_rsrp - 0.5,
    )
    huber_q = torch.where(
        absolute_q <= 1.0, 0.5 * residual_q.square(), absolute_q - 0.5
    )
    count = mask.sum(dim=(1, 2)).clamp_min(1.0)
    return ((huber_rsrp + huber_q) * mask).sum(dim=(1, 2)) / count


def _rotated_offsets(offset, yaw_rad, count, device):
    base = torch.as_tensor(offset, dtype=torch.float32, device=device)
    return base[None, :].expand(count, -1)


def _score_hypothesis(
    model,
    feature_grid,
    geometry,
    template,
    kind_index,
    candidate_xy,
    offset,
    yaw_rad,
    group,
    device,
    ablation,
    candidate_batch=256,
):
    observed_rsrp, observed_q, valid, cw_units = _observations(group, device)
    unit_count = len(group)
    losses = []
    for start in range(0, len(candidate_xy), int(candidate_batch)):
        selected_xy = torch.as_tensor(
            candidate_xy[start : start + int(candidate_batch)],
            dtype=torch.float32,
            device=device,
        )
        count = len(selected_xy)
        center_z = torch.full(
            (count, 1), float(template.center_height_m), device=device
        )
        centers = torch.cat([selected_xy, center_z], dim=1)
        expanded_centers = centers[:, None, :].expand(-1, unit_count, -1).reshape(-1, 3)
        expanded_offsets = _rotated_offsets(
            offset, yaw_rad, count * unit_count, device
        )
        expanded_yaw = torch.full(
            (count * unit_count,), float(yaw_rad), device=device
        )
        expanded_cw = cw_units[None, :].expand(count, -1).reshape(-1)
        object_features, tag_world = extract_object_path_features(
            expanded_centers,
            expanded_yaw,
            expanded_offsets,
            expanded_cw,
            geometry,
            template,
            device,
        )
        query_xy = tag_world[:, :2] + model.geometry_offset_xy_m()[None, :]
        static_features = feature_grid.query_torch(query_xy, expanded_cw, device)
        prediction = model(
            static_features,
            object_features,
            torch.full(
                (count * unit_count,), int(kind_index), dtype=torch.long, device=device
            ),
            expanded_centers[:, :2],
            expanded_cw,
            ablation=ablation,
        )
        losses.append(
            _energy_per_candidate(
                model, prediction, observed_rsrp, observed_q, valid, count
            ).detach()
        )
    return torch.cat(losses).cpu().numpy()


def score_candidate_surface(
    model,
    feature_grid,
    geometry,
    templates,
    group,
    device,
    yaw_candidates_rad=(0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0),
    ablation="full",
):
    xx, yy = np.meshgrid(feature_grid.x_values_m, feature_grid.y_values_m, indexing="xy")
    candidate_xy = np.column_stack([xx.ravel(), yy.ravel()])
    surfaces = []
    metadata = []
    for kind, kind_index in (
        (ObjectKind.PAPER_BOX.value, 0),
        (ObjectKind.METAL_BOX.value, 1),
    ):
        template = templates[kind]
        tag_id = str(group["tag_id"].iloc[0])
        matching_offsets = np.flatnonzero(template.tag_ids.astype(str) == tag_id)
        offset_indices = matching_offsets.tolist() or list(range(len(template.tag_offsets_m)))
        for yaw in yaw_candidates_rad:
            for offset_index in offset_indices:
                offset = template.tag_offsets_m[offset_index]
                surface = _score_hypothesis(
                    model,
                    feature_grid,
                    geometry,
                    template,
                    kind_index,
                    candidate_xy,
                    offset,
                    yaw,
                    group,
                    device,
                    ablation,
                )
                surfaces.append(surface)
                metadata.append(
                    {
                        "object_kind": kind,
                        "object_kind_index": kind_index,
                        "yaw_rad": float(yaw),
                        "offset_index": int(offset_index),
                    }
                )
    return candidate_xy, np.asarray(surfaces), metadata


def _refine_one(
    initial_xy,
    hypothesis,
    model,
    feature_grid,
    geometry,
    templates,
    group,
    device,
    ablation,
    max_steps,
):
    template = templates[hypothesis["object_kind"]]
    observed_rsrp, observed_q, valid, cw_units = _observations(group, device)
    unit_count = len(group)
    xmin, xmax = float(feature_grid.x_values_m[0]), float(feature_grid.x_values_m[-1])
    ymin, ymax = float(feature_grid.y_values_m[0]), float(feature_grid.y_values_m[-1])
    initial = np.asarray(initial_xy, dtype=np.float64)
    normalized = np.array(
        [
            np.clip((initial[0] - xmin) / max(xmax - xmin, 1.0e-6), 1.0e-4, 1.0 - 1.0e-4),
            np.clip((initial[1] - ymin) / max(ymax - ymin, 1.0e-6), 1.0e-4, 1.0 - 1.0e-4),
        ]
    )
    raw = torch.nn.Parameter(
        torch.as_tensor(np.log(normalized / (1.0 - normalized)), dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.LBFGS(
        [raw], max_iter=int(max_steps), line_search_fn="strong_wolfe"
    )

    def position():
        fraction = torch.sigmoid(raw)
        return torch.stack(
            [xmin + (xmax - xmin) * fraction[0], ymin + (ymax - ymin) * fraction[1]]
        )

    def closure():
        optimizer.zero_grad()
        xy = position()
        center = torch.stack(
            [xy[0], xy[1], torch.tensor(template.center_height_m, device=device)]
        )[None, :].expand(unit_count, -1)
        offset = torch.as_tensor(
            template.tag_offsets_m[hypothesis["offset_index"]],
            dtype=torch.float32,
            device=device,
        )[None, :].expand(unit_count, -1)
        yaw = torch.full((unit_count,), float(hypothesis["yaw_rad"]), device=device)
        object_features, tag_world = extract_object_path_features(
            center, yaw, offset, cw_units, geometry, template, device
        )
        static_features = feature_grid.query_torch(
            tag_world[:, :2] + model.geometry_offset_xy_m()[None, :], cw_units, device
        )
        prediction = model(
            static_features,
            object_features,
            torch.full(
                (unit_count,),
                int(hypothesis["object_kind_index"]),
                dtype=torch.long,
                device=device,
            ),
            center[:, :2],
            cw_units,
            ablation=ablation,
        )
        loss = _energy_per_candidate(
            model, prediction, observed_rsrp, observed_q, valid, 1
        )[0]
        loss.backward()
        return loss

    optimizer.step(closure)
    final_loss = float(closure().detach().cpu().item())
    return position().detach().cpu().numpy(), final_loss


def _posterior(losses, temperature):
    flat = np.asarray(losses, dtype=np.float64).reshape(-1)
    logits = -(flat - np.nanmin(flat)) / max(float(temperature), 1.0e-6)
    logits = np.clip(logits, -80.0, 0.0)
    weights = np.exp(logits)
    return weights / max(float(weights.sum()), 1.0e-12)


def localize_group(
    model,
    feature_grid,
    geometry,
    templates,
    group,
    device,
    temperature=1.0,
    top_k=32,
    refine_steps=50,
    yaw_candidates_rad=(0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0),
    ablation="full",
    training_centers_xy=None,
    return_surface=False,
):
    candidate_xy, surfaces, metadata = score_candidate_surface(
        model,
        feature_grid,
        geometry,
        templates,
        group,
        device,
        yaw_candidates_rad=yaw_candidates_rad,
        ablation=ablation,
    )
    flat_order = np.argsort(surfaces, axis=None)
    best_xy = None
    best_loss = float("inf")
    best_hypothesis = None
    for flat_index in flat_order[: min(int(top_k), len(flat_order))]:
        hypothesis_index, candidate_index = np.unravel_index(flat_index, surfaces.shape)
        hypothesis = metadata[hypothesis_index]
        if refine_steps > 0:
            refined_xy, refined_loss = _refine_one(
                candidate_xy[candidate_index],
                hypothesis,
                model,
                feature_grid,
                geometry,
                templates,
                group,
                device,
                ablation,
                refine_steps,
            )
        else:
            refined_xy = candidate_xy[candidate_index]
            refined_loss = float(surfaces[hypothesis_index, candidate_index])
        if refined_loss < best_loss:
            best_xy = refined_xy
            best_loss = refined_loss
            best_hypothesis = hypothesis

    probabilities = _posterior(surfaces, temperature)
    repeated_centers = np.tile(candidate_xy, (len(metadata), 1))
    distances = np.linalg.norm(repeated_centers - best_xy[None, :], axis=1)
    order = np.argsort(distances)
    cumulative = np.cumsum(probabilities[order])
    radius_index = min(int(np.searchsorted(cumulative, 0.9)), len(order) - 1)
    r90 = float(distances[order[radius_index]])
    object_posterior = {}
    probability_matrix = probabilities.reshape(surfaces.shape)
    for kind in (ObjectKind.PAPER_BOX.value, ObjectKind.METAL_BOX.value):
        selected = [index for index, item in enumerate(metadata) if item["object_kind"] == kind]
        object_posterior[kind] = float(probability_matrix[selected].sum())
    template = templates[best_hypothesis["object_kind"]]
    offset = template.tag_offsets_m[best_hypothesis["offset_index"]]
    cosine = math.cos(best_hypothesis["yaw_rad"])
    sine = math.sin(best_hypothesis["yaw_rad"])
    rotated_offset = np.array(
        [
            cosine * offset[0] - sine * offset[1],
            sine * offset[0] + cosine * offset[1],
            offset[2],
        ]
    )
    center_z = float(template.center_height_m)
    tag_position = np.array([best_xy[0], best_xy[1], center_z]) + rotated_offset
    nearest_training = None
    if training_centers_xy is not None and len(training_centers_xy):
        nearest_training = float(
            np.min(np.linalg.norm(np.asarray(training_centers_xy) - best_xy[None, :], axis=1))
        )
    result = {
        "pred_x": float(best_xy[0]),
        "pred_y": float(best_xy[1]),
        "pred_z": center_z,
        "tag_x": float(tag_position[0]),
        "tag_y": float(tag_position[1]),
        "tag_z": float(tag_position[2]),
        "object_kind": best_hypothesis["object_kind"],
        "object_kind_paper_probability": object_posterior[ObjectKind.PAPER_BOX.value],
        "object_kind_metal_probability": object_posterior[ObjectKind.METAL_BOX.value],
        "yaw_rad": float(best_hypothesis["yaw_rad"]),
        "loss": float(best_loss),
        "r90_m": r90,
        "p_within_1m": float(probabilities[distances <= 1.0].sum()),
        "p_within_3m": float(probabilities[distances <= 3.0].sum()),
        "p_within_5m": float(probabilities[distances <= 5.0].sum()),
        "nearest_training_center_m": nearest_training,
        "ood_flag": bool(nearest_training is not None and nearest_training > 3.0),
    }
    if return_surface:
        result["_surface"] = {
            "candidate_xy": candidate_xy,
            "losses": surfaces,
            "metadata": metadata,
        }
    return result


def threshold_metrics(rows):
    errors = np.asarray([row["error_m"] for row in rows], dtype=np.float64)
    if not len(errors):
        return {"count": 0}
    r90_covered = [row["error_m"] <= row["r90_m"] for row in rows]
    return {
        "count": int(len(errors)),
        "mean_m": float(errors.mean()),
        "median_m": float(np.median(errors)),
        "p90_m": float(np.quantile(errors, 0.9)),
        "max_m": float(errors.max()),
        "success_at_1m": float(np.mean(errors <= 1.0)),
        "success_at_3m": float(np.mean(errors <= 3.0)),
        "success_at_5m": float(np.mean(errors <= 5.0)),
        "r90_coverage": float(np.mean(r90_covered)),
    }


def calibrate_temperature(surface_records, candidates=(0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)):
    best_temperature = 1.0
    best_nll = float("inf")
    for temperature in candidates:
        nll = 0.0
        for record in surface_records:
            surface = record["_surface"]
            probabilities = _posterior(surface["losses"], temperature)
            centers = np.tile(surface["candidate_xy"], (len(surface["metadata"]), 1))
            distance = np.linalg.norm(centers - record["truth_xy"][None, :], axis=1)
            mass = float(probabilities[distance <= 0.20].sum())
            nll -= math.log(max(mass, 1.0e-12))
        if nll < best_nll:
            best_nll = nll
            best_temperature = float(temperature)
    return best_temperature


def evaluate_split(
    frame,
    split,
    model,
    feature_grid,
    geometry,
    templates,
    device,
    temperature,
    ablation,
    top_k,
    refine_steps,
    training_centers_xy,
    return_surfaces=False,
):
    rows = []
    subset = frame[frame["split"] == split]
    for parent_id, group in subset.groupby("parent_group_id", sort=True):
        result = localize_group(
            model,
            feature_grid,
            geometry,
            templates,
            group,
            device,
            temperature=temperature,
            top_k=top_k,
            refine_steps=refine_steps,
            ablation=ablation,
            training_centers_xy=training_centers_xy,
            return_surface=return_surfaces,
        )
        truth = group[["object_center_x_map", "object_center_y_map"]].iloc[0].to_numpy(float)
        result.update(
            {
                "parent_group_id": int(parent_id),
                "tag_id": str(group["tag_id"].iloc[0]),
                "true_x": float(truth[0]),
                "true_y": float(truth[1]),
                "true_object_kind": str(group["object_kind"].iloc[0]),
                "error_m": float(np.linalg.norm(np.array([result["pred_x"], result["pred_y"]]) - truth)),
            }
        )
        if return_surfaces:
            result["truth_xy"] = truth
        rows.append(result)
    return rows
