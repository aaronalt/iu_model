import unittest

from sqlalchemy import create_engine, MetaData
from database import Data, RegressionException
import pandas as pd
import numpy as np
from model import Model


class DatabaseTest(unittest.TestCase):

    engine = create_engine("sqlite:///python_models.db", echo=True)
    meta = MetaData()

    def setUp(self):
        self.df = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                               columns=['a', 'b', 'c'])
        self._data = Data("unittest")
        self._data.add_test_table('unittest')

    def test_csv_to_df(self):
        self.df.to_csv('./tests/unittest.csv')
        self._data.csv_to_df()
        self.assertEqual(type(pd.DataFrame()), type(self._data.df))

    def test_csv_to_db(self):
        self.assertTrue(self._data.csv_to_db())

    def test_fit_model(self):
        with self.assertRaises(RegressionException):
            self._data.fit_model(1, Data('ideal'), 'real')
        self.df = pd.DataFrame([[1., 2.], [4., 5.]], columns=['x', 'y1'])
        self.df.to_csv('./tests/unittest.csv')
        self._data.csv_to_df()
        model = self._data.fit_model(1, None, 'linear')
        self.assertEqual(type(model), type(Model(self.df['x'], self.df['y1'], 1)))

    def tearDown(self):
        try:
            self._data.drop_test_table()
        except AttributeError:
            print('table not dropped')


if __name__ == '__main__':
    unittest.main()
