import unittest
import numpy as np
import pandas as pd

from model import Model
from visualize import Graph, PlotTypeException


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
        self.testGraph = Graph('train', df=self.df)
        self.subdir = './tests'

        self.m1 = Model(self.df['x'], self.df['y1'], 1, rss=0.02345, order=1)
        self.m2 = Model(self.df['x'], self.df['y2'], 1, rss=0.02345, order=2)
        self.m3 = Model(self.df['x'], self.df['y3'], 1, rss=0.12345, order=3)
        self.m4 = Model(self.df['x'], self.df['y4'], 1, rss=0.12345, order=4)
        self.m1d = {'y1': self.m1, 'y2': self.m2, 'y3': self.m3, 'y4': self.m4}
        self.m2d = {'y1': self.m1, 'y2': self.m2, 'y3': self.m3, 'y4': self.m4}
        self.m3d = {'y1': self.m1, 'y2': self.m2, 'y3': self.m3, 'y4': self.m4}

    def test_make_subplots(self):
        testGraphPlots = self.testGraph.make_subplots(
            self.testGraph.title, subdir=self.subdir
        )
        self.assertTrue(testGraphPlots)
        with open(f'{self.subdir}/{self.testGraph.title}.pdf', 'r') as file:
            self.assertTrue(file, 'file should have been created in (subdir) dir')

        # test model comparison

        testGraphTest = Graph('model', df=self.df)
        tg = testGraphTest.make_subplots('model',
                                         models={'m1': self.m1d, 'm2': self.m2d, 'm3': self.m3d},
                                         subdir=self.subdir)
        self.assertTrue(tg)
        for i in range(1, 4):
            with open(f'{self.subdir}/y{i}_{testGraphTest.title}.pdf', 'r') as file:
                self.assertTrue(file, 'file should have been created in (subdir) dir')

    def test_plot_model(self):
        with self.assertRaises(PlotTypeException):
            self.testGraph.plot_model(self.m1)
        self.assertTrue(self.testGraph.plot_model(
            self.m1, plt_type='best fit', subdir='./tests')
        )

        rand_x, rand_y = np.random.randint(-20, 20, size=400), \
                        np.random.randint(-1000, 1000, size=400)
        df = pd.DataFrame()
        df['x'] = rand_x
        df['y'] = rand_y
        fit_model = Model(df['x'], df['y'], 1)
        testGraph = Graph('idealtest', df=df)
        self.assertTrue(testGraph.plot_model(
            fit_model, plt_type='ideal', subdir='./tests', fit_model=fit_model)
        )


if __name__ == '__main__':
    unittest.main()
