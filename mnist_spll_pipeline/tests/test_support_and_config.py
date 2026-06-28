from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from pipeline1_config import build_pipeline_context
from pipeline2_config import training_paths
from pipeline_support import load_config, save_config


class ConfigSupportTests(unittest.TestCase):
    def test_one_level_config_inheritance_deep_merges_mappings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "base.yaml").write_text(
                yaml.safe_dump(
                    {
                        "seed": 7,
                        "nested": {"keep": 1, "replace": 2},
                        "items": [1, 2],
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            (root / "child.yaml").write_text(
                yaml.safe_dump(
                    {
                        "extends": "base.yaml",
                        "nested": {"replace": 9, "added": 3},
                        "items": [4],
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            config = load_config(root / "child.yaml")

            self.assertEqual(config["seed"], 7)
            self.assertEqual(config["nested"], {"keep": 1, "replace": 9, "added": 3})
            self.assertEqual(config["items"], [4])
            self.assertEqual(config["_base_config_path"], str((root / "base.yaml").resolve()))

    def test_save_config_strips_runtime_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "saved.yaml"
            config = {
                "seed": 42,
                "_config_path": "/tmp/source.yaml",
                "_config_dir": "/tmp",
                "_base_config_path": "/tmp/base.yaml",
            }
            save_config(config, destination)
            payload = yaml.safe_load(destination.read_text(encoding="utf-8"))
            self.assertEqual(payload, {"seed": 42})

    def test_pipeline_path_construction_has_no_directory_side_effects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = {
                "_config_dir": str(root),
                "paths": {"outputs_root": "out", "output_root": "train-out"},
                "inference": {},
            }

            pipeline1 = build_pipeline_context(config)
            pipeline2 = training_paths(config)

            self.assertFalse(pipeline1.paths.experiment_root.exists())
            self.assertFalse(pipeline2.root.exists())
            pipeline1.paths.ensure_stage_dirs()
            pipeline2.ensure_prepare_dirs()
            self.assertTrue(pipeline1.paths.inputs_root.is_dir())
            self.assertTrue(pipeline2.schedules_root.is_dir())
            self.assertFalse(pipeline1.paths.compiled_root.exists())
            self.assertFalse(pipeline2.compiled_root.exists())

    def test_visualization_imports_do_not_load_torch(self) -> None:
        code = """
import json, sys
tracked = ['torch', 'torchvision']
before = {name: name in sys.modules for name in tracked}
import visualize_results, visualize_spll_training
after = {name: name in sys.modules for name in tracked}
print(json.dumps({'before': before, 'after': after}))
"""
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=PIPELINE_DIR,
            text=True,
            capture_output=True,
            check=True,
        )
        loaded = json.loads(completed.stdout.strip())
        self.assertEqual(loaded["after"], loaded["before"])

    def test_diagnostic_compatibility_facades_keep_required_imports(self) -> None:
        from mnist_spll_common import (
            get_model_variants,
            get_variant_model_output_path,
            load_config as compatibility_load_config,
            resolve_device,
        )
        from mnist_spll_pipeline_core import (
            build_pipeline_context as compatibility_build_context,
            build_read_mnist,
            load_staged_experiments,
        )

        self.assertIs(compatibility_load_config, load_config)
        for value in (
            get_model_variants,
            get_variant_model_output_path,
            resolve_device,
            compatibility_build_context,
            build_read_mnist,
            load_staged_experiments,
        ):
            self.assertTrue(callable(value))

    def test_dispatcher_and_compile_imports_are_lightweight(self) -> None:
        code = """
import json, sys
tracked = ['torch', 'torchvision', 'matplotlib', 'numpy']
before = {name: name in sys.modules for name in tracked}
import run_spll_pipeline, run_spll_training_pipeline, compile_spll, compile_spll_training
after = {name: name in sys.modules for name in tracked}
print(json.dumps({'before': before, 'after': after}))
"""
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=PIPELINE_DIR,
            text=True,
            capture_output=True,
            check=True,
        )
        loaded = json.loads(completed.stdout.strip())
        self.assertEqual(loaded["after"], loaded["before"])
        self.assertFalse(loaded["after"]["torch"] or loaded["after"]["torchvision"] or loaded["after"]["matplotlib"])


if __name__ == "__main__":
    unittest.main()
