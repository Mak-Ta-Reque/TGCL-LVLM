"""Tests for tgcl/device_utils.py."""

import os
import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

from tgcl.device_utils import DeviceConfig
from tgcl.device_utils import get_device_config
from tgcl.device_utils import parse_device_config
from tgcl.device_utils import resolve_device_env


class TestParseDeviceConfig(unittest.TestCase):
    """Test parse_device_config for all supported DEVICE formats."""

    def test_cpu(self):
        dc = parse_device_config("cpu")
        self.assertEqual(dc.device_map, "cpu")
        self.assertIsNone(dc.max_memory)
        self.assertEqual(str(dc.primary_device), "cpu")
        self.assertEqual(dc.gpu_ids, [])
        self.assertTrue(dc.is_cpu_only)
        self.assertFalse(dc.is_single_gpu)
        self.assertFalse(dc.is_multi_gpu)

    @patch("tgcl.device_utils.torch")
    def test_none_defaults_to_auto(self, mock_torch):
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 2
        dc = parse_device_config(None)
        self.assertEqual(dc.raw, "auto")
        self.assertEqual(dc.device_map, "auto")

    @patch("tgcl.device_utils.torch")
    def test_auto_string(self, mock_torch):
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 2
        dc = parse_device_config("auto")
        self.assertEqual(dc.device_map, "auto")
        self.assertEqual(dc.raw, "auto")

    @patch("tgcl.device_utils.torch")
    def test_auto_without_gpu_is_cpu(self, mock_torch):
        mock_torch.cuda.is_available.return_value = False
        mock_torch.cuda.device_count.return_value = 0
        dc = parse_device_config("auto")
        self.assertEqual(dc.raw, "auto")
        self.assertEqual(dc.device_map, "cpu")

    @patch("tgcl.device_utils.torch")
    def test_single_gpu_cuda_1(self, mock_torch):
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 4
        props = MagicMock()
        props.total_mem = 24 * (1024**3)  # 24 GiB
        mock_torch.cuda.get_device_properties.return_value = props
        mock_torch.device.side_effect = lambda x: MagicMock(
            __str__=lambda _: x, type="cuda", index=1
        )

        dc = parse_device_config("cuda:1")
        self.assertEqual(dc.device_map, "auto")
        self.assertIn(1, dc.max_memory)
        self.assertIn("cpu", dc.max_memory)
        self.assertEqual(dc.gpu_ids, [1])
        self.assertTrue(dc.is_single_gpu)

    @patch("tgcl.device_utils.torch")
    def test_multi_gpu_bracket_syntax(self, mock_torch):
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 4
        props = MagicMock()
        props.total_mem = 24 * (1024**3)
        mock_torch.cuda.get_device_properties.return_value = props
        mock_torch.device.side_effect = lambda x: MagicMock(
            __str__=lambda _: x, type="cuda", index=0
        )

        dc = parse_device_config("cuda:[0,2]")
        self.assertEqual(dc.device_map, "auto")
        self.assertIn(0, dc.max_memory)
        self.assertIn(2, dc.max_memory)
        self.assertIn("cpu", dc.max_memory)
        self.assertEqual(dc.gpu_ids, [0, 2])
        self.assertTrue(dc.is_multi_gpu)

    @patch("tgcl.device_utils.torch")
    def test_multi_gpu_comma_syntax(self, mock_torch):
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 4
        props = MagicMock()
        props.total_mem = 24 * (1024**3)
        mock_torch.cuda.get_device_properties.return_value = props
        mock_torch.device.side_effect = lambda x: MagicMock(
            __str__=lambda _: x, type="cuda", index=0
        )

        dc = parse_device_config("cuda:0,2")
        self.assertEqual(dc.device_map, "auto")
        self.assertIn(0, dc.max_memory)
        self.assertIn(2, dc.max_memory)
        self.assertEqual(dc.gpu_ids, [0, 2])
        self.assertTrue(dc.is_multi_gpu)

    @patch("tgcl.device_utils.torch")
    def test_cuda_no_index(self, mock_torch):
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 2
        props = MagicMock()
        props.total_mem = 16 * (1024**3)
        mock_torch.cuda.get_device_properties.return_value = props
        mock_torch.device.side_effect = lambda x: MagicMock(
            __str__=lambda _: x, type="cuda", index=0
        )

        dc = parse_device_config("cuda")
        self.assertEqual(dc.device_map, "auto")
        self.assertEqual(dc.gpu_ids, [0])
        self.assertTrue(dc.is_single_gpu)

    @patch("tgcl.device_utils.torch")
    def test_invalid_gpu_fallback_cpu(self, mock_torch):
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1  # only GPU 0 exists
        mock_torch.device.side_effect = lambda x: MagicMock(
            __str__=lambda _: x, type="cpu"
        )

        dc = parse_device_config("cuda:5")
        # GPU 5 doesn't exist, should fall back to CPU
        self.assertEqual(dc.device_map, "cpu")
        self.assertEqual(dc.gpu_ids, [])

    def test_unrecognised_string_is_rejected(self):
        # A typo must not silently run everything on the CPU.
        with self.assertRaises(ValueError):
            parse_device_config("something_weird")


class TestResolveDeviceEnv(unittest.TestCase):

    def test_default_auto(self):
        with patch.dict(os.environ, {}, clear=True):
            # Remove DEVICE if present
            os.environ.pop("DEVICE", None)
            self.assertEqual(resolve_device_env(), "auto")

    def test_reads_env(self):
        with patch.dict(os.environ, {"DEVICE": "cuda:2"}):
            self.assertEqual(resolve_device_env(), "cuda:2")


class TestGetDeviceConfig(unittest.TestCase):

    def test_override_wins(self):
        dc = get_device_config("cpu")
        self.assertEqual(dc.device_map, "cpu")

    def test_env_fallback(self):
        with patch.dict(os.environ, {"DEVICE": "cpu"}):
            dc = get_device_config(None)
            self.assertEqual(dc.device_map, "cpu")


class TestDeviceConfigProperties(unittest.TestCase):
    """Primary device and GPU flags of a DeviceConfig."""

    def test_cpu_only(self):
        dc = DeviceConfig(gpu_ids=[])
        self.assertTrue(dc.is_cpu_only)
        self.assertFalse(dc.is_single_gpu)
        self.assertFalse(dc.is_multi_gpu)

    def test_single_gpu(self):
        dc = DeviceConfig(gpu_ids=[0])
        self.assertFalse(dc.is_cpu_only)
        self.assertTrue(dc.is_single_gpu)
        self.assertFalse(dc.is_multi_gpu)

    def test_multi_gpu(self):
        dc = DeviceConfig(gpu_ids=[0, 2])
        self.assertFalse(dc.is_cpu_only)
        self.assertFalse(dc.is_single_gpu)
        self.assertTrue(dc.is_multi_gpu)


if __name__ == "__main__":
    unittest.main()
