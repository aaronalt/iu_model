import unittest

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

    # def test_match_ideal_functions(self):



if __name__ == '__main__':
    unittest.main()
