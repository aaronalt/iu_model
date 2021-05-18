import unittest
import numpy as np
import pandas as pd

from visualize import Graph


class VisualizeTest(unittest.TestCase):

    def setUp(self):
        rand_x, rand_y = np.random.randint(-20, 20, size=400), \
                         np.random.randint(-1000, 1000, size=400)
        self.df = pd.DataFrame()
        self.df['x'] = rand_x
        self.df['y1'] = rand_y
        self.df['y2'] = rand_y
        self.df.sort_values(by=['x', 'y1', 'y2'], inplace=True)
        self.testGraph = Graph('train', self.df)

    def test_make_subplots(self):
        subdir = './tests'
        self.assertTrue(self.testGraph.make_subplots(
            self.testGraph.title, subdir=subdir
        ))
        with open(f'{subdir}/{self.testGraph.title}.pdf', 'r') as file:
            self.assertTrue(file, 'file should have been created in (subdir) dir')


if __name__ == '__main__':
    unittest.main()
