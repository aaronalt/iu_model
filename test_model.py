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

    def test_match_ideal_functions(self):
        with open('./datasets/ideal.csv', 'r') as csv:
            with open('./datasets/train.csv', 'r') as train:
                idealData = Data('ideal', _create=False)
                ideal_funs = idealData.csv_to_df()
                ideal_funs.set_index('x')

                # build train_master df
                trainDf = pd.read_csv(train)
                for i in range(1, 4):
                    _if = f'y{i}_if'
                    _ifN = f'y{i+1}'
                    _max = f'y{i}_max_err'
                    _maxN = np.random.randint(0.0001, 999.9999, size=400)
                    _bf = f'y{i}_best_fit'
                    _bfN = f'y{i+10}'
                    trainDf[_if] = _ifN
                    trainDf[_max] = _maxN
                    trainDf[_bf] = _bfN

                newTestModel = self.testModel.df.reset_index(drop=True)
                self.testModel.__setattr__('df', newTestModel)
                ideal_funs = ideal_funs.set_index('x')

                for row in newTestModel.itertuples(index=False):
                    print(row.x, row.y)
                    for c in ideal_funs.keys():
                        print(c)
                        print(f'taking RMSE of:\n{[row.y]} '
                              f'and {[ideal_funs.at[row.x, c]]}')

                newTest = self.testModel.match_ideal_functions(
                    ideal_funs, trainDf, self.m1d, map_train=False
                )
                self.assertTrue(newTest)


if __name__ == '__main__':
    unittest.main()
