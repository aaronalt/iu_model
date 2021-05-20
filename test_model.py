import unittest

from sklearn.metrics import mean_squared_error

from database import Data
from model import Model
import pandas as pd
import numpy as np


class ModelTest(unittest.TestCase):

    def setUp(self):
        rand_x, rand_y = np.random.randint(-20, 20, size=400), \
                         np.random.randint(-1000, 1000, size=400)
        self.df = pd.DataFrame()
        self.df['x'] = rand_x
        self.df['y'] = rand_y
        self.df.sort_values(by=['x', 'y'], inplace=True)
        self.testModel = Model(self.df['x'], self.df['y'], 1, df=self.df)

        rand_x, rand_y = np.random.randint(-20, 20, size=400), \
                         np.random.randint(-1000, 1000, size=400)
        self.dfTest = pd.DataFrame()
        self.dfTest['x'] = rand_x
        self.dfTest['y1'] = rand_y
        self.dfTest['y2'] = rand_y
        self.dfTest['y3'] = rand_y
        self.dfTest['y4'] = rand_y
        self.dfTest.sort_values(by=['x', 'y1', 'y2', 'y3', 'y4'], inplace=True)
        self.subdir = './tests'

        self.m1 = Model(self.dfTest['x'], self.dfTest['y1'], 1, rss=0.02345, order=1)
        self.m2 = Model(self.dfTest['x'], self.dfTest['y2'], 1, rss=0.02345, order=2)
        self.m3 = Model(self.dfTest['x'], self.dfTest['y3'], 1, rss=0.12345, order=3)
        self.m4 = Model(self.dfTest['x'], self.dfTest['y4'], 1, rss=0.12345, order=4)
        self.m1d = {'y1': self.m1, 'y2': self.m2, 'y3': self.m3, 'y4': self.m4}
        self.m2d = {'y1': self.m1, 'y2': self.m2, 'y3': self.m3, 'y4': self.m4}
        self.m3d = {'y1': self.m1, 'y2': self.m2, 'y3': self.m3, 'y4': self.m4}

    def test_find_ideal_function(self):
        with open('./datasets/ideal.csv', 'r') as csv:
            idealData = Data('ideal', _create=False)
            idealData.csv_to_df()
            self.assertEqual(self.testModel.__getattribute__('rmse'), 1000,
                             'rmse should be at the default 1000')
            self.testModel.find_ideal_function(idealData)
            self.assertNotEqual(self.testModel.__getattribute__('rmse'), 1000,
                                'rmse should change after function runs')
            self.assertTrue(self.testModel.__getattribute__('ideal_col'),
                            'ideal_col should change after function runs')
            self.assertNotEqual(self.testModel.__getattribute__('max_dev'), 1000,
                                'max_dev should change after function runs')
            self.assertNotEqual(self.testModel.df.size, 0,
                                'Model() df should be init')


if __name__ == '__main__':
    unittest.main()
