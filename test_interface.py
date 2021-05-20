import unittest

from interface import Interface, NSizeError


class InterfaceTest(unittest.TestCase):

    def setUp(self):
        self._n = {
            'y1': [5, 21, 36],
            'y2': [5, 22, 36],
            'y4': [3, 9, 27]
        }

    def test_training_subplots(self):
        self.assertTrue(Interface(plot_training_subplots=True,
                                  continue_matching=False))

    def test_without_plot(self):
        df = Interface(plot=False,
                       map_train=False,
                       _n=self._n,
                       to_db=False,
                       create_tables=False)
        self.assertEqual(type(df), type(df),
                         'should return pandas df')

    def test_with_plot(self):
        df = Interface(map_train=False,
                       _n=self._n,
                       to_db=False,
                       create_tables=False)
        self.assertEqual(type(df), type(df),
                         'should return pandas df')

    def test_n_size(self):
        n = {
            'y1': [5, 21, 36],
            'y2': [5, 22, 36],
            'y4': [3, 9, 27, 4]
        }
        with self.assertRaises(NSizeError):
            df = Interface(map_train=False,
                           _n=n,
                           to_db=False,
                           create_tables=False)


if __name__ == '__main__':
    unittest.main()
