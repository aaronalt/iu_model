"""
Aaron Althauser
IU International University of Applied Sciences
Data Science, M.Sc.
"""

from database import Data
from model import Model
from visualize import Graph
import pandas as pd


def map_and_compare(*args):
    """returns either self.df and models, or only self.df without running Model functions"""
    return args


class Interface:
    """


    """

    def __init__(self, **kwargs):
        """
        :keyword map_train: run Model functions entirely and
        plot matched ideal vs test data

        :keyword to_db: create SQLite db table for created Data obj

        :keyword plt_type: which type of fit algorithm to run, 'linear' or 'best fit'

        :keyword with_rmse: include rmse values in graph

        :keyword print_table: save stats from model comparisons as a .pdf table

        :keyword plot: plot data and save

        :keyword plot_training_subplots: display and save training data as subplots

        :keyword compare_models: shortcut to just plot comparison of fitted models,
        values is dict of fitted model dicts
        """

        to_db = kwargs.get('to_db', True)
        _create = kwargs.get('create_tables', True)

        self.train = Data('training_data', to_db=to_db)
        self.ideal = Data('ideal_functions', to_db=to_db)
        self.test = Data('test_data')

        self.train_master = self.train.csv_to_df()
        self.train_graph = Graph("Training Data", df=self.train.csv_to_df())
        self.ideal_fn_dict = {'x': self.train.df['x']}

        self._n = kwargs.get('_n', {})
        self.plt_type = kwargs.get('plt_type', 'best fit')
        self.with_rmse = kwargs.get('with_rmse', True)
        self.print_table = kwargs.get('print_table', True)
        self.plot = kwargs.get('plot', True)

        map_train = kwargs.get('map_train', True)
        continue_matching = kwargs.get('continue_matching', True)

        self.models = dict()

        global model

        if 'compare_models' in kwargs.keys():
            models = kwargs.get('compare_models')
            self.train_graph.make_subplots('Model Comparison',
                                           models={'m1': models['m1'],
                                                   'm2': models['m2'],
                                                   'm3': models['m3']})

        if 'plot_training_subplots' in kwargs.keys():
            self.train_graph.make_subplots(self.train_graph.title)

        if continue_matching:

            while self._n['y1']:
                n = {'y1': self._n['y1'].pop(),
                     'y2': self._n['y2'].pop(),
                     'y4': self._n['y4'].pop()}
                self._fit(n)

            self.ideal_fn_df = pd.DataFrame(data=self.ideal_fn_dict)
            self.ideal_fn_df = self.ideal_fn_df.set_index('x')

            test_df = self.test.csv_to_df()
            test_model = Model(test_df['x'], test_df['y'], 1, df=test_df)

            finals = test_model.match_ideal_functions(self.ideal_fn_df,
                                                      self.train_master,
                                                      self.models,
                                                      map_train=map_train)

            if 'run_complete' in kwargs.keys():
                self.test.df_to_db(finals[0])
                map_and_compare(finals)
            else:
                map_and_compare(finals)

    def _fit(self, n):

        for i in range(1, 5):

            col = f'y{i}'
            _if, _max, _bf = f'y{i}_if', f'y{i}_max_err', f'y{i}_best_fit'

            if i != 3:
                model = self.train.fit_model(
                    i, self.ideal, 'poly.fit',
                    order=n[col], print_table=self.print_table)
            else:
                model = self.train.fit_model(
                    i, self.ideal, 'linear',
                    print_table=self.print_table)
            print(f'{col}: ideal_col_array:\n{model.ideal_col_array}')
            self.models[col] = model
            self.ideal_fn_dict[model.ideal_col] = model.ideal_col_array
            self.train_master[_if] = model.ideal_col
            self.train_master[_max] = model.max_dev
            self.train_master[_bf] = model

            if self.plot:
                self.train_graph.plot_model(model,
                                            plt_type=self.plt_type,
                                            with_rmse=self.with_rmse)
