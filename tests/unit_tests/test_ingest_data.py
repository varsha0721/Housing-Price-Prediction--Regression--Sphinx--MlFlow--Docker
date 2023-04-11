import sys
import unittest

sys.path.insert(0, "../TCE_2023")

from ingest_data import add


class TestIngest_data(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(10, 5), 15)
        self.assertEqual(add(-1, 1), 0)
        self.assertEqual(add(-1, -1), -2)


if __name__ == "__main__":
    unittest.main()
