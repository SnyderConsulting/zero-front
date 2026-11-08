import copy
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

from training.selfplay import (ActorCritic, Opponents, SCALARS, TENSORS, cpu_state,
                               initialize_actor, parse_args, publish, read_state,
                               save_state, set_training_phase, training_config)


class LatestRoutingRandom:
    def __init__(self):
        self.choices = 0

    def random(self):
        return 0.0

    def randrange(self, count):
        self.choices += 1
        return (self.choices - 1) % count

    def choice(self, entries):
        return entries[-1]


class SelfPlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(19)
        self.args = parse_args(["--device", "cpu", "--pool-size", "2"])
        self.model = ActorCritic()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.rng = random.Random(7)
        self.pool = Opponents(torch.device("cpu"), self.rng, 2, .5)
        self.pool.add(cpu_state(self.model), 0)
        self.identity = {"runId": "selfplay-test", "createdAt": "2026-11-07T00:00:00Z", "sourceSha256": "test-source"}
        self.counters = {"updates": 1, "actorSteps": 64, "learnerSteps": 48,
                         "completedMatches": 0, "wallSeconds": 2.0}
        self.config = training_config(self.args)

    @staticmethod
    def optimization_step(model, optimizer):
        # Use random data and every actor/critic parameter, so an absent Adam
        # moment or wrong RNG state changes the next update observably.
        pixels = torch.randint(0, 256, (2, 36, 64, 3), dtype=torch.uint8)
        scalars = torch.rand((2, 6))
        action, logprob, entropy, value = model.act(pixels, scalars)
        loss = (logprob.square() + value.square() - entropy * .01).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return action.detach().clone(), cpu_state(model)

    def save(self, directory):
        save_state(directory, self.model, self.optimizer, self.pool, self.rng,
                   self.config, self.identity, self.counters)

    def initial_policy(self, directory, **training):
        path = directory / "initial-policy.json"
        state = cpu_state(self.model)
        artifact = {"format": "arena-pixel-policy", "version": 1, "name": "Teacher student",
                    "createdAt": "2026-11-07T00:00:00Z", "architecture": "cnn-v2",
                    "observation": {"kind": "semantic-rgb", "width": 64, "height": 36, "scalars": SCALARS},
                    "actionSpace": "discrete-v2", "actionRepeat": 6,
                    "weights": {name: {"shape": shape, "data": state[key].flatten().tolist()}
                                for name, (key, shape) in TENSORS.items()},
                    "training": {"algorithm": "dagger", "teacher": "scripted", "samples": 200,
                                 "trainSeeds": [100, 101], "priorTrainingSeeds": [98, 99],
                                 "validationSeeds": [200, 201], "selectionSeeds": [300], **training}}
        path.write_text(json.dumps(artifact), encoding="utf-8")
        return path

    def test_imitation_actor_initializes_exactly_with_fresh_critic_and_optimizer(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            initial = self.initial_policy(directory)
            args = parse_args(["--device", "cpu", "--initial", str(initial)])
            config = training_config(args)
            fresh = ActorCritic()
            optimizer = torch.optim.Adam(fresh.parameters(), lr=args.learning_rate)
            identity = initialize_actor(fresh, args, config)
            for _, (key, _) in TENSORS.items():
                torch.testing.assert_close(fresh.state_dict()[key], self.model.state_dict()[key], rtol=0, atol=0)
            self.assertEqual(torch.count_nonzero(fresh.value.weight).item(), 0)
            self.assertEqual(torch.count_nonzero(fresh.value.bias).item(), 0)
            self.assertEqual(len(optimizer.state), 0)
            self.assertEqual(identity["origin"], "teacher-initialized")
            self.assertEqual(identity["initialTraining"]["trainingSeedRanges"], [[98, 102]])
            self.assertEqual(identity["initialModelSha256"], config["initial"]["sha256"])

    def test_teacher_checkpoint_preserves_lineage_and_rejects_changed_initial_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            initial = self.initial_policy(directory)
            self.args = parse_args(["--device", "cpu", "--pool-size", "2", "--initial", str(initial)])
            self.config = training_config(self.args)
            self.identity.update(initialize_actor(self.model, self.args, self.config))
            self.save(directory)
            saved = read_state(directory, self.config, "test-source")
            self.assertEqual(saved["initialTraining"], self.identity["initialTraining"])
            self.assertEqual(saved["initialModelSha256"], self.identity["initialModelSha256"])
            with self.assertRaisesRegex(ValueError, "initialization mode"):
                read_state(directory, training_config(parse_args(["--device", "cpu", "--pool-size", "2"])), "test-source")
            initial.write_text(initial.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "training settings changed"):
                read_state(directory, training_config(self.args), "test-source")
            with self.assertRaisesRegex(ValueError, "changed while starting"):
                initialize_actor(ActorCritic(), self.args, self.config)

    def test_teacher_initialization_protects_benchmark_and_validation_seed_boundaries(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for metadata, expected in (({"priorTrainingSeeds": [600002]}, "benchmark seeds"),
                                       ({"priorValidationSeeds": [1000001]}, "validation/selection"),
                                       ({"selectionSeeds": [600003]}, "benchmark seeds"),
                                       ({"algorithm": "ppo-strict-self-play"}, "imitation-trained")):
                with self.subTest(metadata=metadata):
                    initial = self.initial_policy(directory, **metadata)
                    args = parse_args(["--device", "cpu", "--initial", str(initial)])
                    with self.assertRaisesRegex(ValueError, expected):
                        initialize_actor(ActorCritic(), args, training_config(args))

    def test_critic_warmup_keeps_actor_and_vision_exactly_fixed(self):
        args = parse_args(["--device", "cpu", "--critic-warmup-updates", "2"])
        self.assertTrue(set_training_phase(self.model, args, 0))
        before = cpu_state(self.model)
        pixels = torch.randint(0, 256, (4, 36, 64, 3), dtype=torch.uint8)
        scalars = torch.rand((4, 6))
        self.optimizer.zero_grad(set_to_none=True)
        self.model(pixels, scalars)[1].sub(1).square().mean().backward()
        self.optimizer.step()
        for _, (key, _) in TENSORS.items():
            torch.testing.assert_close(self.model.state_dict()[key], before[key], rtol=0, atol=0)
        self.assertFalse(torch.equal(self.model.value.bias, before["value.bias"]))
        self.assertFalse(set_training_phase(self.model, args, 2))
        self.assertTrue(all(parameter.requires_grad for parameter in self.model.parameters()))

    def test_published_teacher_model_reports_ancestry_and_selfplay_phase_truthfully(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            initial = self.initial_policy(directory)
            args = parse_args(["--device", "cpu", "--initial", str(initial), "--output", str(directory / "live.json")])
            identity = {**self.identity, **initialize_actor(self.model, args, training_config(args))}
            observation = {"pixels": np.zeros((1, 36, 64, 3), dtype=np.uint8), "scalars": np.zeros((1, 6), dtype=np.float32)}
            output, parity = publish(self.model, args, identity, self.counters, observation, directory)
            training = json.loads(output.read_text(encoding="utf-8"))["training"]
            self.assertEqual(parity["parity"], "passed")
            self.assertEqual(training["algorithm"], "ppo-self-play-after-imitation")
            self.assertTrue(training["demonstrations"])
            self.assertFalse(training["selfPlayDemonstrations"])
            self.assertFalse(training["scriptedTrainingOpponents"])
            self.assertEqual(training["priorTrainingSeedRanges"], [[98, 102]])
            self.assertEqual(training["validationSeedRanges"], [[200, 202]])
            self.assertEqual(training["selectionSeedRanges"], [[300, 301]])

    def test_checkpoint_resumes_adam_and_random_sequence(self):
        self.optimization_step(self.model, self.optimizer)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.save(directory)
            expected_route_random = self.rng.random()
            expected_actions, expected_weights = self.optimization_step(self.model, self.optimizer)
            saved = read_state(directory, self.config, "test-source")
            restored = ActorCritic()
            restored.load_state_dict(saved["model"])
            optimizer = torch.optim.Adam(restored.parameters(), lr=self.args.learning_rate)
            optimizer.load_state_dict(saved["optimizer"])
            self.assertGreater(len(optimizer.state), 0)
            self.rng.setstate(saved["routingRandomState"])
            torch.set_rng_state(saved["torchRandomState"])
            self.assertEqual(self.rng.random(), expected_route_random)
            actual_actions, actual_weights = self.optimization_step(restored, optimizer)
            torch.testing.assert_close(actual_actions, expected_actions, rtol=0, atol=0)
            for key in expected_weights:
                torch.testing.assert_close(actual_weights[key], expected_weights[key], rtol=0, atol=0)
            self.assertEqual(saved["counters"], self.counters)

    def test_snapshot_creation_does_not_consume_training_rng(self):
        before = torch.get_rng_state().clone()
        self.pool.add(cpu_state(self.model), 1)
        torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)

    def test_reward_revision_history_survives_checkpoint_roundtrip(self):
        self.identity["rewardHistory"] = [{"fromProfile": "shaped-v1", "toProfile": "shaped-v2",
                                          "drawBefore": 0, "drawAfter": -.5, "update": 1}]
        self.identity["rewardSpec"] = {"profile": "shaped-v2", "draw": -.5}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.save(directory)
            saved = read_state(directory, self.config, "test-source")
            self.assertEqual(saved["rewardHistory"], self.identity["rewardHistory"])
            self.assertEqual(saved["rewardSpec"], self.identity["rewardSpec"])
            self.assertEqual(saved["counters"], self.counters)

    def test_frozen_teams_stay_fixed_until_each_episode_ends(self):
        routing_random = LatestRoutingRandom()
        pool = Opponents(torch.device("cpu"), routing_random, 2, 1.0)
        pool.add(cpu_state(self.model), 0)
        observation = {"count": 8, "teams": np.array([0, 0, 1, 1] * 2),
                       "environments": np.array([0] * 4 + [1] * 4),
                       "episodeIds": np.array([0] * 8)}
        learner, frozen = pool.route(observation)
        np.testing.assert_equal(learner, [0, 0, 1, 1, 1, 1, 0, 0])
        first_policies = [policy for _, policy in frozen]
        pool.add(cpu_state(self.model), 1)
        pool.add(cpu_state(self.model), 2)
        again, frozen_again = pool.route(observation)
        np.testing.assert_equal(again, learner)
        self.assertTrue(all(policy is first for (_, policy), first in zip(frozen_again, first_policies)))
        self.assertEqual(routing_random.choices, 2)
        observation["episodeIds"][:4] = 1
        _, after_reset = pool.route(observation)
        self.assertIs(after_reset[0][1], pool.pool[-1][1])
        self.assertIs(after_reset[1][1], first_policies[1])
        self.assertEqual([update for update, _ in pool.pool], [0, 2])

    def test_current_versus_current_learns_from_both_teams(self):
        self.pool.probability = 0
        observation = {"count": 4, "teams": np.array([0, 0, 1, 1]),
                       "environments": np.zeros(4, dtype=np.int64),
                       "episodeIds": np.zeros(4, dtype=np.int64)}
        learner, frozen = self.pool.route(observation)
        np.testing.assert_equal(learner, np.ones(4))
        self.assertEqual(frozen, [])

    def test_resume_rejects_non_selfplay_origin_and_changed_configuration(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.save(directory)
            original = torch.load(directory / "checkpoint.pt", weights_only=True)
            for key, value in (("origin", "imitation"), ("format", "unrelated-checkpoint"), ("sourceSha256", "different-source")):
                altered = copy.deepcopy(original)
                altered[key] = value
                torch.save(altered, directory / "checkpoint.pt")
                with self.assertRaisesRegex(ValueError, "Resume rejected"):
                    read_state(directory, self.config, "test-source")
            torch.save(original, directory / "checkpoint.pt")
            with self.assertRaisesRegex(ValueError, "training settings changed"):
                read_state(directory, {**self.config, "seed": self.args.seed + 1}, "test-source")
            with self.assertRaisesRegex(ValueError, "training settings changed"):
                read_state(directory, {**self.config, "reward_profile": "sparse-v1"}, "test-source")

    def test_corrupt_latest_checkpoint_falls_back_to_previous_save(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.save(directory)
            self.counters = {**self.counters, "updates": 2}
            self.save(directory)
            invalid = torch.load(directory / "checkpoint.pt", weights_only=True)
            invalid["model"]["policy.weight"][0, 0] = float("nan")
            torch.save(invalid, directory / "checkpoint.pt")
            saved = read_state(directory, self.config, "test-source")
            self.assertEqual(saved["counters"]["updates"], 1)

    def test_parameters_reject_nonfinite_values_and_benchmark_seed_overlap(self):
        cases = [["--entropy", "nan"], ["--target-kl", "inf"], ["--save-minutes", "nan"],
                 ["--snapshot-minutes", "inf"], ["--benchmark-minutes", "-1"],
                 ["--critic-warmup-updates", "-1"],
                 ["--seed", "-1"], ["--name", ""], ["--seed-start", "600000"]]
        for arguments in cases:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                parse_args(arguments)


if __name__ == "__main__":
    unittest.main()
