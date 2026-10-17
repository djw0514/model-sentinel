import unittest

from model_sentinel.pack import load_pack
from model_sentinel.sampling import sample_order


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.cases = load_pack()["cases"]

    def test_stratified_round_contains_every_category(self):
        ordered, seed = sample_order(self.cases, seed=17)
        categories = {case["category"] for case in self.cases}
        self.assertEqual(seed, 17)
        for index in range(3):
            self.assertEqual({case["category"] for case in ordered[index * 5:(index + 1) * 5]}, categories)
        self.assertEqual({case["id"] for case in ordered}, {case["id"] for case in self.cases})
        self.assertEqual(len(ordered), len(self.cases))

    def test_seed_replays_and_different_seed_rotates_probes(self):
        first, _ = sample_order(self.cases, seed=17)
        repeat, _ = sample_order(self.cases, seed=17)
        different, _ = sample_order(self.cases, seed=18)
        self.assertEqual(first, repeat)
        self.assertNotEqual(first[:12], different[:12])
        self.assertEqual(self.cases[0]["id"], "arith-01")

    def test_generated_seed_is_recordable_and_reproducible(self):
        ordered, seed = sample_order(self.cases)
        replay, _ = sample_order(self.cases, seed=seed)
        self.assertEqual(ordered, replay)

    def test_ordered_strategy_preserves_pack_order(self):
        ordered, _ = sample_order(self.cases, strategy="ordered", seed=1)
        self.assertEqual(ordered, self.cases)
        self.assertIsNot(ordered, self.cases)

    def test_invalid_strategy_and_seed_are_rejected(self):
        for options in ({"strategy": "random"}, {"seed": True}, {"seed": 1.0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                sample_order(self.cases, **options)


if __name__ == "__main__":
    unittest.main()
