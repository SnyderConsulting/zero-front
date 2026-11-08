import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np


def isolated_trainer():
    directory = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location('_expert_test_base', directory / 'train.py')
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    spec = importlib.util.spec_from_file_location('_expert_test_trainer', directory / 'train-expert.py')
    trainer = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'train': base}):
        spec.loader.exec_module(trainer)
    return trainer


class ExpertDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trainer = isolated_trainer()

    @staticmethod
    def samples(*_):
        return (np.zeros((2, 36, 64, 3), dtype=np.uint8),
                np.zeros((2, 6), dtype=np.float32), np.zeros((2, 6), dtype=np.int64))

    def test_replay_records_seed_ancestry_and_cannot_leak_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            common = ['--validation-seeds', '20', '--cache-dir', directory,
                      '--max-samples', '2', '--max-validation-samples', '2']
            args = self.trainer.parser().parse_args([*common, '--train-seeds', '10'])
            with patch.object(self.trainer, 'collect', side_effect=self.samples):
                _, _, config = self.trainer.load_dataset(args)
            replay = config['datasetCache']
            with self.assertRaisesRegex(ValueError, 'overlap'):
                self.trainer.load_replay(replay, [10])
            args = self.trainer.parser().parse_args([*common, '--train-seeds', '11', '--replay-data', replay])
            with patch.object(self.trainer, 'collect', side_effect=self.samples):
                train, valid, config = self.trainer.load_dataset(args)
            self.assertEqual(config['priorTrainingSeeds'], [10])
            self.assertEqual(len(train[0]), 4)
            self.assertEqual(len(valid[0]), 2)
            self.assertEqual(config['replayDatasets'][0]['samples'], 2)
            # Reuse the cache only after exact provenance comparison.
            with patch.object(self.trainer, 'collect', side_effect=AssertionError('Should use cache')):
                self.trainer.load_dataset(args)

    def test_replay_without_provenance_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'old.npz'
            np.savez(path, **dict(zip(('train_pixels', 'train_scalars', 'train_labels'), self.samples())))
            with self.assertRaisesRegex(ValueError, 'provenance'):
                self.trainer.load_replay(path, [20])

    def test_engine_changes_invalidate_collection_source_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = ('dist/simulation.js', 'dist/renderer.js', 'dist/mapgen.js', 'dist/policy.js',
                     'training/expert.mjs', 'training/train-expert.py')
            for name in files:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(name)
            with patch.object(self.trainer.base, 'ROOT', root):
                before = self.trainer.dataset_source_hash()
                (root / 'dist/renderer.js').write_text('changed pixels')
                self.assertNotEqual(before, self.trainer.dataset_source_hash())

    def test_native_scripted_teacher_cannot_silently_supply_broken_dagger_labels(self):
        args = self.trainer.parser().parse_args(['--scripted-teacher', '--learner', 'unused.json'])
        with self.assertRaisesRegex(ValueError, 'demonstration collection only'):
            self.trainer.load_dataset(args)

    def test_collection_partition_keeps_serial_per_seed_quota_and_output_order(self):
        args = self.trainer.parser().parse_args(['--collection-workers', '4', '--max-samples', '17'])
        seeds = [10, 11, 12, 13, 14]
        self.assertEqual(self.trainer.collection_groups(seeds, 17, 4),
                         [([10, 11], 6), ([12], 3), ([13], 3), ([14], 3)])
        def worker(group, _args, _split, budget, _processes):
            if group[0] == 10:
                time.sleep(.02)  # Complete later, while retaining original seed order.
            array = np.repeat(group, budget // len(group))
            return array, array, array
        with patch.object(self.trainer, 'collect_worker', side_effect=worker):
            result = self.trainer.collect(seeds, args, 'train')
        for array in result:
            np.testing.assert_array_equal(array, np.repeat(seeds, 3))

    def test_serial_and_parallel_node_collection_are_identical(self):
        args = self.trainer.parser().parse_args(['--train-seeds', '10:14', '--seconds', '1',
            '--team-size', '2', '--actors-per-frame', '2', '--max-samples', '21'])
        serial = self.trainer.collect(args.train_seeds, args, 'train')
        args.collection_workers = 4
        parallel = self.trainer.collect(args.train_seeds, args, 'train')
        self.assertEqual(len(serial[0]), 20)
        for first, second in zip(serial, parallel):
            np.testing.assert_array_equal(first, second)

    def test_failed_worker_terminates_and_reaps_running_peers(self):
        args = self.trainer.parser().parse_args(['--collection-workers', '2', '--max-samples', '4'])
        started, terminated = threading.Event(), threading.Event()
        children = []
        class Child:
            def __init__(self, *_args, **_options):
                self.returncode = None
                self.waited = False
                children.append(self)
            def poll(self):
                return self.returncode
            def terminate(self):
                self.returncode = 1
                terminated.set()
            def wait(self, timeout=None):
                self.waited = True
                return self.returncode
        def worker(group, _args, _split, _budget, processes):
            if group[0] == 10:
                process = processes.start(['unused'])
                started.set()
                self.assertTrue(terminated.wait(3), 'Peer was not terminated after worker failure')
                processes.finished(process)
                return self.samples()
            self.assertTrue(started.wait(3))
            raise RuntimeError('simulated collection failure')
        with patch.object(self.trainer.subprocess, 'Popen', Child), patch.object(self.trainer, 'collect_worker', side_effect=worker):
            with self.assertRaisesRegex(RuntimeError, 'simulated collection failure'):
                self.trainer.collect([10, 11], args, 'train')
        self.assertEqual(len(children), 1)
        self.assertEqual(children[0].returncode, 1)
        self.assertTrue(children[0].waited)


if __name__ == '__main__':
    unittest.main()
