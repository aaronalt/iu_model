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
        'y1': [1, 20, 36],
        'y2': [1, 20, 36],
        'y4': [3, 9, 27]
    }

    df = Interface(_n=_n,
                   to_db=True,  # set to 'False' for future iterations
                   create_tables=False,
                   plot_training_subplots=True,
                   plot_order_error=True)

    df2 = Interface(continue_matching=False,
                    compare_models={'m1': df.models_master_1,
                                    'm2': df.models_master_2,
                                    'm3': df.models_master_3})

    '''
    What did we discover from 1st iteration:
    
    method:
    - y1: set lowest, middle, and highest polynomial to check on the graph
    - y2: same
    - y3: no n order
    - y4: from looking at the graph we know its a cubic function, so an n order of 3 will
    more than likely be the best fit. However, we dont know if there will be a higher chance
    of minimizing least-squares by changing certain parameters and hyperparameters
    
    observations:
    - fn(y1) could use a lower-order polynomial for a more general fit without overfitting
      - test: lower order vs. 1st best fit
      - n-17: rss cuts in half, rmse reduces a lot
      - n-19: rss halfs again, rmse reduces, AND MRE more than halfs
    - fn(y2): 
      - n-18: rss halfs, rmse halfs, mre cuts little
      - n-20: rss halfs, rmse halfs, mre cuts
      - n-20 - n-32: no drastic changes in rss, rmse climbs, mre stable
    - fn(y3): needs linear and not polynomial function to reduce rss
    - fn(y4): best order is 3, but can rss be reduced? still too large 
      - n-3: rss obliterates, but rmse jumps to 102.41, mre jumps to 289
      - n-9: rss down a litte, rmse down more, mre at highest
    '''


# todo: experiment with different weights, programmatically (for paper, what to do next)


if __name__ == "__main__":
    main()
