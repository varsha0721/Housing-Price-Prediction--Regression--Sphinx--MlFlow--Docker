import sys
import unittest

sys.path.insert(0, "../TCE_2023")

from train import subtract


class TestTrain(unittest.TestCase):
    def test_subtract(self):
        self.assertEqual(subtract(10, 5), 5)
        self.assertEqual(subtract(-1, 1), -2)
        self.assertEqual(subtract(-1, -1), 0)


if __name__ == "__main__":
    unittest.main()
