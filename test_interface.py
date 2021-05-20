import unittest

from interface import Interface


class InterfaceTest(unittest.TestCase):

    def test_training_subplots(self):
        self.assertTrue(Interface(plot_training_subplots=True,
                                  continue_matching=False,
                                  drop_tables_after=True))

    def test_without_plot(self):
        _n = {
            'y1': [5, 21, 36],
            'y2': [5, 22, 36],
            'y4': [3, 9, 27]
        }
        df = Interface(plot=False,
                       map_train=False,
                       _n=_n,
                       to_db=False,
                       create_tables=False)
        self.assertEqual(type(df), type(df),
                         'should return pandas df')


if __name__ == '__main__':
    unittest.main()
