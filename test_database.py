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
        self.df = pd.DataFrame([[1.5555555, 2.55555555], [4.5555555, 5.5555555]],
                               columns=['x', 'y1'])
        self.df.to_csv('./tests/unittest.csv')
        self._data.csv_to_df()
        with self.assertRaises(Exception):
            self._data.fit_model(1, None, 'linear')
        # test print_table arg
        with open('./datasets/ideal.csv', 'r') as idf:
            ideal = Data('ideal', _create=False)
            ideal.csv_to_df()
            self.assertFalse(ideal.is_empty(), 'df obj should be populated')
            model = self._data.fit_model(1, ideal, 'linear',
                                         print_table=True, table_name='./tests/unittest')
            self.assertEqual(type(model), type(Model(self.df['x'], self.df['y1'], 1)))
            self.assertEqual(model.x[0], self.df['x'][0])

    def tearDown(self):
        try:
            self._data.drop_test_table()
        except AttributeError:
            print('table not dropped')


if __name__ == '__main__':
    unittest.main()
