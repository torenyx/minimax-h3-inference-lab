"""Check published-file selection and the sample's preparation/launch chain."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from download_models import BASE_REPO, PROFILES, complete, download_plan
from sample import commands_for


class ProfileDownloads(unittest.TestCase):
    def test_preallocated_partial_file_is_resumed(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'model.safetensors'
            path.write_bytes(b'abc')
            item = {'destination': str(path), 'size': 3}
            self.assertTrue(complete(item))
            Path(str(path) + '.aria2').touch()
            self.assertFalse(complete(item))

    def files(self, repo, revision):
        if repo == BASE_REPO:
            return [{'rfilename': name, 'size': 10} for name in [
                'LICENSE', 'FL2VA/transformer/config.json', 'Ref2VA/transformer/config.json', 'assets/demo.mp4']]
        return [{'rfilename': value[3], 'size': 20} for value in PROFILES.values() if value[1] == repo]

    def test_downloads_only_requested_variant_and_exact_adapter(self):
        for profile, spec in PROFILES.items():
            with self.subTest(profile=profile):
                plan = download_plan(profile, Path('/models'), fetch=self.files)
                self.assertEqual([item['file'] for item in plan], ['LICENSE', spec[0] + '/transformer/config.json', spec[3]])
                self.assertTrue(all(item['revision'] in item['url'] for item in plan))

    def test_missing_adapter_fails_before_download(self):
        with self.assertRaisesRegex(ValueError, 'missing'):
            download_plan('ref2va-pdd', Path('/models'), fetch=lambda repo, rev: self.files(repo, rev) if repo == BASE_REPO else [])

    def test_existing_checkpoint_skips_conversion_and_preserves_reference_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths, _, commands = commands_for('ref2va-pdd', root, 4, 124, root / 'output')
            self.assertEqual(len(commands), 2)
            self.assertIn('Ref2VA', commands[0])
            checkpoint = Path(paths['pruned_checkpoint'])
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()
            _, _, commands = commands_for('ref2va-pdd', root, 4, 124, root / 'output')
            self.assertEqual(len(commands), 1)
            self.assertIn('ref2va-pdd', commands[0])
            self.assertTrue(any(item.endswith('ref2va-reference.png') for item in commands[0]))


if __name__ == '__main__':
    unittest.main()
