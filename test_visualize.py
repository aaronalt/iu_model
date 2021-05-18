import unittest
import numpy as np
import pandas as pd

from model import Model
from visualize import Graph


class VisualizeTest(unittest.TestCase):

    def setUp(self):
        rand_x, rand_y = np.random.randint(-20, 20, size=400), \
                         np.random.randint(-1000, 1000, size=400)
        self.df = pd.DataFrame()
        self.df['x'] = rand_x
        self.df['y1'] = rand_y
        self.df['y2'] = rand_y
        self.df['y3'] = rand_y
        self.df['y4'] = rand_y
        self.df.sort_values(by=['x', 'y1', 'y2', 'y3', 'y4'], inplace=True)
        self.testGraph = Graph('train', self.df)

    def test_make_subplots(self):
        subdir = './tests'
        testGraphPlots = self.testGraph.make_subplots(
            self.testGraph.title, subdir=subdir
        )
        self.assertTrue(testGraphPlots)
        with open(f'{subdir}/{self.testGraph.title}.pdf', 'r') as file:
            self.assertTrue(file, 'file should have been created in (subdir) dir')

        # test model comparison
        m1 = Model(self.df['x'], self.df['y1'], 1, rss=0.02345, order=1)
        m2 = Model(self.df['x'], self.df['y2'], 1, rss=0.02345, order=2)
        m3 = Model(self.df['x'], self.df['y3'], 1, rss=0.12345, order=3)
        m4 = Model(self.df['x'], self.df['y4'], 1, rss=0.12345, order=4)
        m1d = {'y1': m1, 'y2': m2, 'y3': m3, 'y4': m4}
        m2d = {'y1': m1, 'y2': m2, 'y3': m3, 'y4': m4}
        m3d = {'y1': m1, 'y2': m2, 'y3': m3, 'y4': m4}

        testGraphTest = Graph('model', self.df)
        tg = testGraphTest.make_subplots('model',
                                         models={'m1': m1d, 'm2': m2d, 'm3': m3d},
                                         subdir=subdir)
        self.assertTrue(tg)
        for i in range(1, 4):
            with open(f'{subdir}/y{i}_{testGraphTest.title}.pdf', 'r') as file:
                self.assertTrue(file, 'file should have been created in (subdir) dir')



if __name__ == '__main__':
    unittest.main()
