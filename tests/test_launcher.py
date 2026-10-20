"""CPU-only checks for task contracts and the generated launch specification."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime.checkpoint_names import normalize_final_norm

source = Path(__file__).resolve().parents[1] / 'scripts/run.py'
loader = importlib.util.spec_from_file_location('h3_launcher', source)
launcher = importlib.util.module_from_spec(loader)
loader.loader.exec_module(launcher)


class LaunchContract(unittest.TestCase):
    def test_historical_final_norm_name_preserves_tensor(self):
        value = object()
        weights = normalize_final_norm({'final_layer.norm.weight': value})
        self.assertEqual(set(weights), {'norm_out.norm.weight'})
        self.assertIs(weights['norm_out.norm.weight'], value)
        self.assertIs(normalize_final_norm(weights)['norm_out.norm.weight'], value)

    def arguments(self, profile, extra=()):
        return launcher.parser().parse_args(['--profile', profile, '--model-dir', 'model',
            '--pruned-checkpoint', 'pruned.safetensors', '--adapter', 'adapter.safetensors',
            '--prompt-file', 'prompt.txt', '--output-dir', 'output', *extra])

    def test_larry_and_ref_turbo_keep_distinct_scales(self):
        larry = launcher.specification(self.arguments('fl2va-turbo', ['--image', 'first.png', '--gpus', '4']))
        ref = launcher.specification(self.arguments('ref2va-turbo', ['--image', 'ref.png']))
        self.assertEqual((larry['steps'], larry['alpha'], larry['height'], larry['width']), (6, 64, 768, 1344))
        self.assertEqual((ref['steps'], ref['alpha']), (8, 8))
        self.assertEqual(larry['gpus'], 4)

    def test_pdd_supports_multiple_modalities(self):
        spec = launcher.specification(self.arguments('ref2va-pdd',
            ['--image', 'a.png', '--image', 'b.png', '--video', 'v.mp4', '--audio', 'a.wav']))
        self.assertEqual(len(spec['image']), 2)
        self.assertEqual(spec['alpha'], 64)

    def test_missing_condition_is_rejected(self):
        for profile in ['fl2va-turbo', 'ref2va-pdd', 'ref2va-turbo']:
            with self.subTest(profile=profile), self.assertRaises(ValueError):
                launcher.specification(self.arguments(profile))

    def test_text_only_and_first_last_frame(self):
        spec = launcher.specification(self.arguments('fl2va-turbo', ['--task', 't2av']))
        self.assertEqual(spec['image'], [])
        with self.assertRaises(ValueError):
            launcher.specification(self.arguments('fl2va-turbo', ['--task', 'fl2av', '--image', 'a.png']))


if __name__ == '__main__':
    unittest.main()
