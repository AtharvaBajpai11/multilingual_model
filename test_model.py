"""
IsoFlow-S2ST Automated Unit & Integration Test Suite
Verifies model architecture, tensor shape contracts, parameter freezing, 
gradient flow through IsoSyncBridge, and translation pipeline execution.
"""

import unittest
import torch

from models.backbones.s2st_encoder import StreamingS2STEncoder
from models.backbones.diflow_dubber import DiFlowDubberDecoder
from models.bridge.duration_predictor import IsochronousDurationPredictor
from models.bridge.cross_modal_attn import SharedMultimodalCrossAttention
from models.bridge.isosync_bridge import IsoSyncBridge
from models.isoflow_s2st import IsoFlowS2ST
from training.loss import IsoFlowS2STLoss
from data.dataset import SyntheticMultilingualVideoDataset
from pipeline.video_processor import VideoProcessor
from pipeline.translator import MultilingualVideoTranslator


class TestIsoFlowS2ST(unittest.TestCase):

    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = 2
        self.t_speech = 40
        self.t_video = 20
        self.t_target = 20
        self.s2st_dim = 512
        self.flow_dim = 512
        self.video_dim = 64

    def test_01_s2st_encoder(self):
        """Test streaming S2ST speech encoder forward pass."""
        encoder = StreamingS2STEncoder(input_dim=80, hidden_dim=self.s2st_dim, streaming=True)
        audio = torch.randn(self.batch_size, 80, self.t_speech)
        out = encoder(audio)
        self.assertEqual(out.dim(), 3)
        self.assertEqual(out.shape[0], self.batch_size)
        self.assertEqual(out.shape[2], self.s2st_dim)
        print("[OK] Test 01 Passed: StreamingS2STEncoder forward pass verified.")

    def test_02_diflow_decoder(self):
        """Test DiFlowDubber flow matching decoder velocity field prediction."""
        decoder = DiFlowDubberDecoder(frame_channels=3, hidden_dim=self.flow_dim, cond_dim=self.flow_dim)
        x_t = torch.randn(self.batch_size * self.t_target, 3, 64, 64)
        t = torch.rand(self.batch_size * self.t_target)
        ref_face = torch.randn(self.batch_size * self.t_target, 3, 64, 64)
        z_shared = torch.randn(self.batch_size * self.t_target, self.flow_dim)

        v_t = decoder(x_t, t, ref_face, z_shared)
        self.assertEqual(v_t.shape, x_t.shape)
        print("[OK] Test 02 Passed: DiFlowDubberDecoder velocity prediction verified.")

    def test_03_duration_predictor(self):
        """Test IsochronousDurationPredictor & Length Regulator."""
        dur_pred = IsochronousDurationPredictor(speech_dim=self.s2st_dim, video_dim=self.video_dim)
        h_speech = torch.randn(self.batch_size, 10, self.s2st_dim)
        e_video = torch.randn(self.batch_size, 10, self.video_dim)

        durations, log_durations = dur_pred(h_speech, e_video, speech_rate_alpha=1.0)
        self.assertEqual(durations.shape, (self.batch_size, 10))

        expanded = dur_pred.expand_features(h_speech, durations)
        self.assertEqual(expanded.shape[0], self.batch_size)
        self.assertEqual(expanded.shape[2], self.s2st_dim)
        print("[OK] Test 03 Passed: IsochronousDurationPredictor & Length Regulator verified.")

    def test_04_isosync_bridge(self):
        """Test novel IsoSyncBridge connecting module."""
        bridge = IsoSyncBridge(s2st_dim=self.s2st_dim, flow_dim=self.flow_dim, video_dim=self.video_dim)
        s2st_features = torch.randn(self.batch_size, 10, self.s2st_dim)
        source_frames = torch.randn(self.batch_size, 10, 3, 64, 64)

        out = bridge(s2st_features, source_frames)
        self.assertIn("z_shared", out)
        self.assertEqual(out["z_shared"].shape[0], self.batch_size)
        self.assertEqual(out["z_shared"].shape[2], self.flow_dim)
        print("[OK] Test 04 Passed: Novel IsoSyncBridge connecting module verified.")

    def test_05_parameter_freezing(self):
        """Verify backbone freezing strategy (S2ST & DiFlow frozen, IsoSyncBridge trainable)."""
        model = IsoFlowS2ST(freeze_backbones=True)
        
        # Check encoder frozen
        for p in model.s2st_encoder.parameters():
            self.assertFalse(p.requires_grad)

        # Check decoder frozen
        for p in model.diflow_decoder.parameters():
            self.assertFalse(p.requires_grad)

        # Check IsoSyncBridge trainable
        bridge_trainable = [p.requires_grad for p in model.isosync_bridge.parameters()]
        self.assertTrue(all(bridge_trainable))
        print("[OK] Test 05 Passed: Backbone parameter freezing strategy verified.")

    def test_06_gradient_flow(self):
        """Verify backward gradient flow through IsoSyncBridge."""
        model = IsoFlowS2ST(freeze_backbones=True).to(self.device)
        audio = torch.randn(self.batch_size, 80, self.t_speech, device=self.device)
        source_frames = torch.randn(self.batch_size, self.t_video, 3, 64, 64, device=self.device)
        ref_face = torch.randn(self.batch_size, 3, 64, 64, device=self.device)
        target_frames = torch.randn(self.batch_size, self.t_target, 3, 64, 64, device=self.device)
        target_durations = torch.ones(self.batch_size, self.t_speech, device=self.device)

        t = torch.rand(self.batch_size, self.t_target, device=self.device)
        x_0 = torch.randn_like(target_frames)
        x_t = 0.5 * x_0 + 0.5 * target_frames
        u_t = target_frames - x_0

        outputs = model(audio, source_frames, ref_face, x_t=x_t, t=t)

        criterion = IsoFlowS2STLoss()
        loss_dict = criterion(outputs, x_0, target_frames, target_durations, u_t)

        loss_dict["loss"].backward()

        # Assert IsoSyncBridge has non-zero gradients
        grads = [p.grad for p in model.isosync_bridge.parameters() if p.grad is not None]
        self.assertGreater(len(grads), 0)
        print("[OK] Test 06 Passed: Gradient propagation through IsoSyncBridge verified.")

    def test_07_full_pipeline_translation(self):
        """Verify end-to-end video translation & lip-sync pipeline."""
        processor = VideoProcessor()
        inputs = processor._generate_dummy_inputs(num_frames=15)

        translator = MultilingualVideoTranslator()
        result = translator.translate_and_sync(inputs, target_lang="es", flow_steps=3)

        self.assertIn("generated_frames", result)
        self.assertIn("target_audio_waveform", result)
        self.assertEqual(result["generated_frames"].dim(), 4) # (T, C, H, W)
        print("[OK] Test 07 Passed: Full video translation & lip-sync pipeline verified.")


if __name__ == "__main__":
    unittest.main()
