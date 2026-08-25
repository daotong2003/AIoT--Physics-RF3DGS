from __future__ import absolute_import

import math

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
except ImportError:  # 允许基础环境读取数据与场景，训练时再检查Torch。
    torch = None
    nn = object
    functional = None


LIGHT_SPEED_M_S = 299792458.0
CARRIER_FREQUENCY_HZ = 895000000.0
CW_POWER_DBM = 33.0
PATH_FEATURE_COUNT = 8


def require_torch():
    if torch is None:
        raise RuntimeError("RF-3DGS训练需要PyTorch，请在WRF-GS服务器环境中运行")


def covariance_precision(covariance_m2, device):
    require_torch()
    covariance = torch.as_tensor(covariance_m2, dtype=torch.float32, device=device)
    eye = torch.eye(3, dtype=covariance.dtype, device=device).unsqueeze(0)
    return torch.linalg.inv(covariance + eye * 1.0e-7)


def gaussian_segment_integral(
    starts,
    ends,
    gaussian_xyz,
    gaussian_precision,
    gaussian_weights,
    segment_chunk=64,
    gaussian_chunk=32768,
):
    """计算各向异性Gaussian沿有限三维线段的解析积分。

    返回每条线段的加权密度积分；支持对线段端点反向传播。
    """
    require_torch()
    if starts.ndim != 2 or starts.shape[1] != 3 or ends.shape != starts.shape:
        raise ValueError("starts/ends必须为[B,3]")
    if gaussian_xyz.ndim != 2 or gaussian_xyz.shape[1] != 3:
        raise ValueError("gaussian_xyz必须为[G,3]")
    if gaussian_precision.shape != (len(gaussian_xyz), 3, 3):
        raise ValueError("gaussian_precision必须为[G,3,3]")
    if gaussian_weights.shape != (len(gaussian_xyz),):
        raise ValueError("gaussian_weights必须为[G]")

    outputs = []
    sqrt_half = math.sqrt(0.5)
    sqrt_pi_over_two = math.sqrt(math.pi / 2.0)
    for segment_start in range(0, len(starts), int(segment_chunk)):
        left = starts[segment_start : segment_start + int(segment_chunk)]
        right = ends[segment_start : segment_start + int(segment_chunk)]
        direction = right - left
        length = torch.linalg.norm(direction, dim=1).clamp_min(1.0e-6)
        total = torch.zeros(len(left), dtype=left.dtype, device=left.device)
        for gaussian_start in range(0, len(gaussian_xyz), int(gaussian_chunk)):
            xyz = gaussian_xyz[gaussian_start : gaussian_start + int(gaussian_chunk)]
            precision = gaussian_precision[
                gaussian_start : gaussian_start + int(gaussian_chunk)
            ]
            weights = gaussian_weights[
                gaussian_start : gaussian_start + int(gaussian_chunk)
            ]
            delta = left[:, None, :] - xyz[None, :, :]
            quadratic = torch.einsum("bi,gij,bj->bg", direction, precision, direction)
            linear = torch.einsum("bi,gij,bgj->bg", direction, precision, delta)
            constant = torch.einsum("bgi,gij,bgj->bg", delta, precision, delta)
            quadratic = quadratic.clamp_min(1.0e-8)
            perpendicular = (constant - linear * linear / quadratic).clamp_min(0.0)
            root = torch.sqrt(2.0 * quadratic)
            lower = linear / root
            upper = (quadratic + linear) / root
            erf_difference = (torch.erf(upper) - torch.erf(lower)).clamp_min(0.0)
            integral = (
                length[:, None]
                * torch.exp(-0.5 * perpendicular).clamp_min(1.0e-30)
                * (sqrt_pi_over_two / torch.sqrt(quadratic))
                * erf_difference
            )
            total = total + torch.sum(integral * weights[None, :], dim=1)
        outputs.append(total)
    return torch.cat(outputs, dim=0)


def _scene_tensors(scene, device):
    xyz = torch.as_tensor(scene.xyz_m, dtype=torch.float32, device=device)
    precision = covariance_precision(scene.covariance_m2, device)
    opacity = torch.as_tensor(scene.opacity, dtype=torch.float32, device=device)
    confidence = torch.as_tensor(
        scene.geometry_confidence, dtype=torch.float32, device=device
    )
    return xyz, precision, opacity, confidence


def extract_static_path_features(
    tag_positions_m,
    cw_ids,
    geometry,
    scene,
    device,
    segment_chunk=64,
    gaussian_chunk=32768,
):
    """为单CW观测生成8条双基地链路的几何与Gaussian路径特征。"""
    require_torch()
    tags = torch.as_tensor(tag_positions_m, dtype=torch.float32, device=device)
    ids = torch.as_tensor(cw_ids, dtype=torch.long, device=device)
    if tags.ndim != 2 or tags.shape[1] != 3 or ids.shape != (len(tags),):
        raise ValueError("tag_positions_m和cw_ids形状不匹配")
    cw_all = torch.as_tensor(geometry.cw_xyz_m, dtype=torch.float32, device=device)
    rx = torch.as_tensor(geometry.rx_xyz_m, dtype=torch.float32, device=device)
    cw = cw_all[ids]
    xyz, precision, opacity, confidence = _scene_tensors(scene, device)
    reliable_weights = opacity * confidence
    uncertain_weights = opacity * (1.0 - confidence)

    forward_reliable = gaussian_segment_integral(
        cw,
        tags,
        xyz,
        precision,
        reliable_weights,
        segment_chunk,
        gaussian_chunk,
    )
    forward_uncertain = gaussian_segment_integral(
        cw,
        tags,
        xyz,
        precision,
        uncertain_weights,
        segment_chunk,
        gaussian_chunk,
    )
    return_starts = tags[:, None, :].expand(-1, 8, -1).reshape(-1, 3)
    return_ends = rx[None, :, :].expand(len(tags), -1, -1).reshape(-1, 3)
    return_reliable = gaussian_segment_integral(
        return_starts,
        return_ends,
        xyz,
        precision,
        reliable_weights,
        segment_chunk,
        gaussian_chunk,
    ).reshape(len(tags), 8)
    return_uncertain = gaussian_segment_integral(
        return_starts,
        return_ends,
        xyz,
        precision,
        uncertain_weights,
        segment_chunk,
        gaussian_chunk,
    ).reshape(len(tags), 8)

    distance_cw = torch.linalg.norm(tags - cw, dim=1).clamp_min(0.1)
    distance_rx = torch.linalg.norm(tags[:, None, :] - rx[None, :, :], dim=2).clamp_min(0.1)
    forward_reliable = forward_reliable[:, None].expand(-1, 8)
    forward_uncertain = forward_uncertain[:, None].expand(-1, 8)
    reflection_forward = torch.log1p(forward_reliable) / distance_cw[:, None]
    reflection_return = torch.log1p(return_reliable) / distance_rx
    return torch.stack(
        [
            distance_cw[:, None].expand(-1, 8),
            distance_rx,
            forward_reliable,
            return_reliable,
            forward_uncertain,
            return_uncertain,
            reflection_forward,
            reflection_return,
        ],
        dim=2,
    )


def transform_world_to_object(points, centers, yaw_rad):
    """将批量世界坐标变换至对应目标的局部坐标。"""
    cosine = torch.cos(yaw_rad)
    sine = torch.sin(yaw_rad)
    delta = points - centers
    x = cosine * delta[..., 0] + sine * delta[..., 1]
    y = -sine * delta[..., 0] + cosine * delta[..., 1]
    return torch.stack([x, y, delta[..., 2]], dim=-1)


def extract_object_path_features(
    object_centers_m,
    yaw_rad,
    tag_offsets_m,
    cw_ids,
    geometry,
    template,
    device,
):
    """动态目标在局部坐标中进行路径积分，中心与偏航角保持可微。"""
    require_torch()
    centers = torch.as_tensor(object_centers_m, dtype=torch.float32, device=device)
    yaw = torch.as_tensor(yaw_rad, dtype=torch.float32, device=device).reshape(-1)
    offsets = torch.as_tensor(tag_offsets_m, dtype=torch.float32, device=device)
    ids = torch.as_tensor(cw_ids, dtype=torch.long, device=device)
    cw_world = torch.as_tensor(geometry.cw_xyz_m, dtype=torch.float32, device=device)[ids]
    rx_world = torch.as_tensor(geometry.rx_xyz_m, dtype=torch.float32, device=device)
    cosine = torch.cos(yaw)
    sine = torch.sin(yaw)
    tag_world = torch.stack(
        [
            centers[:, 0] + cosine * offsets[:, 0] - sine * offsets[:, 1],
            centers[:, 1] + sine * offsets[:, 0] + cosine * offsets[:, 1],
            centers[:, 2] + offsets[:, 2],
        ],
        dim=1,
    )
    cw_local = transform_world_to_object(cw_world, centers, yaw)
    tag_local = offsets
    rx_expanded = rx_world[None, :, :].expand(len(centers), -1, -1)
    centers_expanded = centers[:, None, :].expand_as(rx_expanded)
    yaw_expanded = yaw[:, None].expand(len(centers), 8)
    rx_local = transform_world_to_object(rx_expanded, centers_expanded, yaw_expanded)

    xyz = torch.as_tensor(template.xyz_m, dtype=torch.float32, device=device)
    precision = covariance_precision(template.covariance_m2, device)
    weights = torch.as_tensor(template.opacity, dtype=torch.float32, device=device)
    forward = gaussian_segment_integral(cw_local, tag_local, xyz, precision, weights)
    returns = gaussian_segment_integral(
        tag_local[:, None, :].expand(-1, 8, -1).reshape(-1, 3),
        rx_local.reshape(-1, 3),
        xyz,
        precision,
        weights,
    ).reshape(len(centers), 8)
    return torch.stack([forward[:, None].expand(-1, 8), returns], dim=2), tag_world


def _inverse_softplus(value):
    return math.log(math.expm1(float(value)))


class PhysicalRFModel(nn.Module):
    """受物理约束的链路参数模型，不执行Signal到Position回归。"""

    def __init__(self, cw_count=4, rx_count=8, residual_rank=2):
        require_torch()
        super(PhysicalRFModel, self).__init__()
        self.cw_count = int(cw_count)
        self.rx_count = int(rx_count)
        self.residual_rank = int(residual_rank)
        self.cw_gain_db = nn.Parameter(torch.zeros(self.cw_count))
        self.rx_gain_db = nn.Parameter(torch.zeros(self.rx_count))
        self.tag_reflection_db = nn.Parameter(torch.tensor(-42.0))
        self.noise_floor_dbm = nn.Parameter(torch.full((self.rx_count,), -105.0))
        self.raw_scene_attenuation = nn.Parameter(torch.tensor(_inverse_softplus(1.0)))
        self.raw_reflection_gain = nn.Parameter(torch.tensor(_inverse_softplus(0.5)))
        self.raw_uncertain_scale = nn.Parameter(torch.tensor(_inverse_softplus(0.25)))
        self.raw_paper_attenuation = nn.Parameter(torch.tensor(_inverse_softplus(1.0)))
        self.raw_metal_delta = nn.Parameter(torch.tensor(_inverse_softplus(3.0)))
        self.raw_q_density = nn.Parameter(torch.tensor(_inverse_softplus(0.1)))
        self.log_sigma_rsrp = nn.Parameter(torch.tensor(math.log(4.0)))
        self.log_sigma_q = nn.Parameter(torch.tensor(math.log(4.0)))
        self.residual_basis = nn.Parameter(torch.zeros(6, self.residual_rank))
        self.residual_link = nn.Parameter(
            torch.zeros(self.cw_count, self.rx_count, self.residual_rank)
        )
        # 仅修正点云与设备坐标的全局小偏差，不允许每个高斯自由漂移。
        self.raw_geometry_offset_xy = nn.Parameter(torch.zeros(2))

    def object_attenuation_db_per_unit(self):
        paper = functional.softplus(self.raw_paper_attenuation)
        metal = paper + functional.softplus(self.raw_metal_delta)
        return paper, metal

    def geometry_offset_xy_m(self):
        return 0.03 * torch.tanh(self.raw_geometry_offset_xy)

    def _gaussian_rf_residual(self, static_features, cw_ids):
        """由Gaussian路径属性生成低秩修正，不存储位置到RSRP的指纹。"""
        basis = torch.stack(
            [
                torch.ones_like(static_features[:, :, 0]),
                torch.log1p(static_features[:, :, 2].clamp_min(0.0)),
                torch.log1p(static_features[:, :, 3].clamp_min(0.0)),
                torch.log1p(static_features[:, :, 4].clamp_min(0.0)),
                torch.log1p(static_features[:, :, 5].clamp_min(0.0)),
                torch.log1p(
                    (static_features[:, :, 6] + static_features[:, :, 7]).clamp_min(0.0)
                ),
            ],
            dim=2,
        )
        latent = torch.einsum("bnf,fr->bnr", basis, self.residual_basis)
        link = self.residual_link[cw_ids]
        return torch.sum(latent * link, dim=2)

    def forward(
        self,
        static_features,
        object_features,
        object_kind_index,
        _positions_xy_m,
        cw_ids,
        ablation="full",
    ):
        if static_features.ndim != 3 or static_features.shape[1:] != (
            self.rx_count,
            PATH_FEATURE_COUNT,
        ):
            raise ValueError("static_features必须为[B,8,8]")
        distance_cw = static_features[:, :, 0].clamp_min(0.1)
        distance_rx = static_features[:, :, 1].clamp_min(0.1)
        wavelength = LIGHT_SPEED_M_S / CARRIER_FREQUENCY_HZ
        fspl_cw = 20.0 * torch.log10(4.0 * math.pi * distance_cw / wavelength)
        fspl_rx = 20.0 * torch.log10(4.0 * math.pi * distance_rx / wavelength)
        reliable_density = static_features[:, :, 2] + static_features[:, :, 3]
        uncertain_density = static_features[:, :, 4] + static_features[:, :, 5]
        reflection = static_features[:, :, 6] + static_features[:, :, 7]
        object_density = object_features[:, :, 0] + object_features[:, :, 1]
        paper, metal = self.object_attenuation_db_per_unit()
        object_index = object_kind_index.reshape(-1, 1).to(dtype=torch.bool)
        object_attenuation = torch.where(object_index, metal, paper)
        scene_attenuation = functional.softplus(self.raw_scene_attenuation)
        uncertain_scale = functional.softplus(self.raw_uncertain_scale)
        reflection_gain = functional.softplus(self.raw_reflection_gain)
        residual = self._gaussian_rf_residual(static_features, cw_ids)
        if ablation == "free_space":
            reliable_density = torch.zeros_like(reliable_density)
            uncertain_density = torch.zeros_like(uncertain_density)
            reflection = torch.zeros_like(reflection)
            object_density = torch.zeros_like(object_density)
            residual = torch.zeros_like(residual)
        elif ablation == "static":
            object_density = torch.zeros_like(object_density)
            residual = torch.zeros_like(residual)
        elif ablation == "static_dynamic":
            residual = torch.zeros_like(residual)
        elif ablation != "full":
            raise ValueError("未知消融模式: %s" % ablation)
        direct_dbm = (
            CW_POWER_DBM
            - fspl_cw
            - fspl_rx
            + self.tag_reflection_db
            + self.cw_gain_db[cw_ids, None]
            + self.rx_gain_db[None, :]
            - scene_attenuation * reliable_density
            - uncertain_scale * uncertain_density
            - object_attenuation * object_density
            + residual
        )
        # 直达与一次反射在功率域非相干合成，符合窄带RSS观测约束。
        reflected_strength = (reflection_gain * reflection.clamp_min(0.0)).clamp_min(1.0e-12)
        reflected_dbm = (
            CW_POWER_DBM
            - fspl_cw
            - fspl_rx
            + self.tag_reflection_db
            + self.cw_gain_db[cw_ids, None]
            + self.rx_gain_db[None, :]
            - 0.5 * scene_attenuation * reliable_density
            - uncertain_scale * uncertain_density
            - object_attenuation * object_density
            + 10.0 * torch.log10(reflected_strength)
            + residual
        )
        db_to_neper = math.log(10.0) / 10.0
        rsrp = torch.logsumexp(
            torch.stack([direct_dbm, reflected_dbm], dim=0) * db_to_neper,
            dim=0,
        ) / db_to_neper
        q_density = functional.softplus(self.raw_q_density)
        q_dbm = self.noise_floor_dbm[None, :] + q_density * reliable_density
        sinr = rsrp - q_dbm
        return {"rsrp": rsrp, "q": q_dbm, "sinr": sinr}

    def loss(self, prediction, observed_rsrp, observed_q, valid_mask):
        valid = valid_mask.to(dtype=prediction["rsrp"].dtype)
        count = valid.sum().clamp_min(1.0)
        sigma_rsrp = torch.exp(self.log_sigma_rsrp).clamp(0.5, 20.0)
        sigma_q = torch.exp(self.log_sigma_q).clamp(0.5, 20.0)
        rsrp_residual = (prediction["rsrp"] - observed_rsrp) / sigma_rsrp
        q_residual = (prediction["q"] - observed_q) / sigma_q
        data_loss = (
            functional.smooth_l1_loss(
                rsrp_residual * valid, torch.zeros_like(rsrp_residual), reduction="sum"
            )
            + functional.smooth_l1_loss(
                q_residual * valid, torch.zeros_like(q_residual), reduction="sum"
            )
        ) / count
        uncertainty_loss = self.log_sigma_rsrp + self.log_sigma_q
        smooth_regularization = 1.0e-3 * (
            torch.mean(self.residual_basis ** 2) + torch.mean(self.residual_link ** 2)
        )
        paper, metal = self.object_attenuation_db_per_unit()
        material_regularization = 1.0e-4 * (
            (functional.softplus(self.raw_scene_attenuation) - 1.0).square()
            + (functional.softplus(self.raw_reflection_gain) - 0.5).square()
            + (paper - 1.0).square()
            + (metal - 4.0).square()
        )
        geometry_regularization = 1.0e-2 * torch.mean(
            (self.geometry_offset_xy_m() / 0.03).square()
        )
        gain_centering = 1.0e-3 * (
            torch.mean(self.cw_gain_db) ** 2 + torch.mean(self.rx_gain_db) ** 2
        )
        return (
            data_loss
            + uncertainty_loss
            + material_regularization
            + smooth_regularization
            + geometry_regularization
            + gain_centering
        )
