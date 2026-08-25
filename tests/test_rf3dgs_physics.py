import unittest

try:
    import torch
except ImportError:  # 本地基础环境无Torch，云端WRF-GS环境执行本组测试。
    torch = None


@unittest.skipIf(torch is None, "需要PyTorch")
class GaussianPathPhysicsTest(unittest.TestCase):
    def test_segment_integral_is_finite_and_differentiable(self):
        from rf3dgs_localization.physics import gaussian_segment_integral

        starts = torch.tensor([[-1.0, 0.0, 0.0]], requires_grad=True)
        ends = torch.tensor([[1.0, 0.0, 0.0]], requires_grad=True)
        xyz = torch.tensor([[0.0, 0.0, 0.0]])
        precision = torch.eye(3).reshape(1, 3, 3) / (0.1 ** 2)
        weights = torch.ones(1)
        integral = gaussian_segment_integral(starts, ends, xyz, precision, weights)
        self.assertTrue(torch.isfinite(integral).all().item())
        self.assertGreater(float(integral.item()), 0.0)
        integral.sum().backward()
        self.assertIsNotNone(starts.grad)
        self.assertTrue(torch.isfinite(starts.grad).all().item())

    def test_metal_object_attenuation_is_constrained_above_paper(self):
        from rf3dgs_localization.physics import PhysicalRFModel

        model = PhysicalRFModel(cw_count=4, rx_count=8)
        paper, metal = model.object_attenuation_db_per_unit()
        self.assertGreater(float(metal.item()), float(paper.item()))

    def test_forward_model_obeys_basic_power_physics(self):
        from rf3dgs_localization.physics import PhysicalRFModel

        model = PhysicalRFModel(cw_count=4, rx_count=8)
        static = torch.zeros((2, 8, 8), dtype=torch.float32)
        static[0, :, 0:2] = 1.0
        static[1, :, 0:2] = 2.0
        dynamic = torch.zeros((2, 8, 2), dtype=torch.float32)
        kinds = torch.zeros(2, dtype=torch.long)
        positions = torch.zeros((2, 2), dtype=torch.float32)
        cw_ids = torch.zeros(2, dtype=torch.long)
        prediction = model(static, dynamic, kinds, positions, cw_ids, ablation="free_space")
        self.assertTrue(torch.isfinite(prediction["rsrp"]).all().item())
        self.assertGreater(
            float(prediction["rsrp"][0].mean().item()),
            float(prediction["rsrp"][1].mean().item()),
        )

    def test_metal_dynamic_gaussians_transmit_less_than_paper(self):
        from rf3dgs_localization.physics import PhysicalRFModel

        model = PhysicalRFModel(cw_count=4, rx_count=8)
        static = torch.zeros((2, 8, 8), dtype=torch.float32)
        static[:, :, 0:2] = 1.0
        dynamic = torch.ones((2, 8, 2), dtype=torch.float32)
        prediction = model(
            static,
            dynamic,
            torch.tensor([0, 1], dtype=torch.long),
            torch.zeros((2, 2)),
            torch.zeros(2, dtype=torch.long),
            ablation="static_dynamic",
        )
        self.assertGreater(
            float(prediction["rsrp"][0].mean().item()),
            float(prediction["rsrp"][1].mean().item()),
        )


if __name__ == "__main__":
    unittest.main()
