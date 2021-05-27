"""
Aaron Althauser
IU International University of Applied Sciences
Data Science, M.Sc.
"""

from sqlalchemy import create_engine, MetaData
from interface import Interface
import pandas as pd

engine = create_engine("sqlite:///python_models.db", echo=True)
meta = MetaData()
pd.set_option('display.float_format', lambda x: '%.5f' % x)


def main():
    """

    Main entry point of the program.

    Uses keyword args passed to Interface() object to run
    program with specific options.

    """

    # Set polynomial order of the fit function
    # Last number in the list is the chosen model
    n = {
        'y1': [[5, 15, 36], [20, 18, 17], [19, 21, 23]],
        'y2': [[5, 15, 36], [18, 19, 20], [22, 23, 24]],
        'y4': [[1, 2, 3], [3, 9, 27], [7, 8, 9]]
    }

    while len(n['y1']) > 1:

        print(f'len(n) should be > 1: {len(n["y1"])}')

        _n = dict(y1=n['y1'].pop(0),
                  y2=n['y2'].pop(0),
                  y4=n['y4'].pop(0))
        print(f'_n: {_n}')
        df = Interface(_n=_n,
                       print_table=True,
                       plot_order_error=True)
        print(_n)
        df2 = Interface(_n=_n,
                        to_db=False,
                        run_complete=True,
                        compare_models={'m1': df.models_master_1,
                                        'm2': df.models_master_2,
                                        'm3': df.models_master_3})

        print(df2)

    """
    Final iteration
    """

    _n = {
        'y1': n['y1'][0],
        'y2': n['y2'][0],
        'y4': n['y4'][0]
    }

    df = Interface(_n=_n,
                   create_tables=False,
                   print_table=True,
                   plot_training_subplots=True,
                   plot_order_error=True)

    df2 = Interface(_n=_n,
                    to_db=False,
                    run_complete=True,
                    compare_models={'m1': df.models_master_1,
                                    'm2': df.models_master_2,
                                    'm3': df.models_master_3})

    print(df2)


if __name__ == "__main__":
    main()
