"""
Aaron Althauser
IU International University of Applied Sciences
Data Science, M.Sc.
"""

from database import Data
from model import Model
from visualize import Graph
import pandas as pd


class NSizeError(Exception):
    pass


def check_n_size(n):
    size = [len(i) for i in n.values()]
    if len(n) == 3 \
            and len(n) == size[0] \
            and len(n) == size[1] \
            and len(n) == size[2]:
        return True
    else:
        raise NSizeError('_n size does not match')


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
        self.models_master_1 = dict()
        self.models_master_2 = dict()
        self.models_master_3 = dict()
        self.result = tuple()

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

            check_n_size(self._n)
            idx = 1
            while self._n['y1']:
                n = {'y1': self._n['y1'].pop(0),
                     'y2': self._n['y2'].pop(0),
                     'y4': self._n['y4'].pop(0)}
                print(f'n : {n}\nidx : {idx}')
                self._fit(n, idx)
                idx += 1

            self.ideal_fn_df = pd.DataFrame(data=self.ideal_fn_dict)
            self.ideal_fn_df = self.ideal_fn_df.set_index('x')

            self.test_df = self.test.csv_to_df()
            test_model = Model(self.test_df['x'], self.test_df['y'], 1, df=self.test_df)

            finals = test_model.match_ideal_functions(self.ideal_fn_df,
                                                      self.train_master,
                                                      self.models,
                                                      map_train=map_train)

            if 'run_complete' in kwargs.keys():
                self.test.df_to_db(finals[0])
            else:
                self.result = finals

    def _fit(self, n, idx):

        _m = f'm{idx}'
        new_models = dict()

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

            new_models[col] = model
            self.models[col] = model

            print(f'model: {model}\nself.models: {self.models}')
            print(f'i: {i}, order: {model.order}, col: {col}')

            self.ideal_fn_dict[model.ideal_col] = model.ideal_col_array
            self.train_master[_if] = model.ideal_col
            self.train_master[_max] = model.max_dev
            self.train_master[_bf] = model

            if self.plot:
                self.train_graph.plot_model(model,
                                            plt_type=self.plt_type,
                                            with_rmse=self.with_rmse)

        if idx == 1:
            self.models_master_1[_m] = new_models
        if idx == 2:
            self.models_master_2[_m] = new_models
        if idx == 3:
            self.models_master_3[_m] = new_models

        '''if not self.models_master.get('m1', None):
            self.models_master['m1'] = self.models
        elif not self.models_master.get('m2', None):
            print(f'models_master: {self.models_master}')

            self.models_master['m2'] = self.models
        elif not self.models_master.get('m3', None):
            print(f'models_master: {self.models_master}')

            self.models_master['m3'] = self.models'''
