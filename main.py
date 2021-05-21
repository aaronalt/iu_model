"""
Aaron Althauser
IU International University of Applied Sciences
Data Science, M.Sc.
"""

from sqlalchemy import create_engine, MetaData
from database import Data
from interface import Interface
from model import Model
from visualize import Graph
import pandas as pd

engine = create_engine("sqlite:///python_models.db", echo=True)
meta = MetaData()
pd.set_option('display.float_format', lambda x: '%.5f' % x)


def main():
    """Main entry point of the program."""

    # Set polynomial order for
    _n = {
        'y1': [1, 20, 40],
        'y2': [1, 20, 40],
        'y4': [3, 9, 27]
    }

    df = Interface(_n=_n,
                   to_db=True,  # set to 'False' for future iterations
                   create_tables=False,
                   plot_training_subplots=True)

    df2 = Interface(continue_matching=False,
                    compare_models={'m1': df.models_master_1,
                                    'm2': df.models_master_2,
                                    'm3': df.models_master_3})

    '''
    What did we discover from 1st iteration:
    
    - fn(y1) could use a lower-order polynomial for a more general fit without overfitting
      - test: lower order vs. 1st best fit
    - fn(y2): same as y1
    - fn(y3): needs linear and not polynomial function to reduce rss
    - fn(y4): best order is 3, but can rss be reduced? still too large 
    '''


# todo: experiment with different weights, programmatically (for paper, what to do next)


if __name__ == "__main__":
    main()
