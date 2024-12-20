import unittest
from seaborn_defaults import Seafoam


class MyTestCase(unittest.TestCase):

    def test_something(self):
        self.seafoam = Seafoam()
        assert len(self.seafoam.base_grey) == 3


if __name__ == '__main__':
    unittest.main()