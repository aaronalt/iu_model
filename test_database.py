import unittest

from sqlalchemy import create_engine, MetaData, inspect

from database import Data
import pandas as pd
import numpy as np


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
        self._data.csv_to_df('unittest')
        self.assertEqual(type(pd.DataFrame()), type(self._data.df))

    def test_df_to_db(self):
        self.assertFalse(self._data.df_to_db(pd.DataFrame()))

    def tearDown(self):
        try:
            self._data.drop_test_table()
        except AttributeError:
            print('table not dropped')


if __name__ == '__main__':
    unittest.main()
