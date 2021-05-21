"""
Aaron Althauser
IU International University of Applied Sciences
Data Science, M.Sc.
"""

from itertools import starmap
from sklearn.metrics import mean_squared_error, max_error
import pandas as pd
from visualize import Graph
import numpy as np


class Model:
    """Fit model to training data and match with an ideal function"""

    def __init__(self, x, y, col, df=pd.DataFrame(), rss=0, order=1):
        self.x = x
        self.y = y
        self.col = col
        self.rmse = 1000
        self.test_rmse = 0
        self.max_dev = 1000
        self.__max_dev = 1000
        self.rss = rss
        self.order = order
        self.__bf = ""
        self.ideal_col = ""
        self.ideal_col_array = []
        if not df.__sizeof__() == 0:
            self.df = df

    def get_max_dev(self):
        return self.__max_dev

    def set_max_dev(self, val):
        self.__max_dev = val
        self.max_dev = self.__max_dev

    def get_best_fit(self):
        return self.__bf

    def set_best_fit(self, val):
        self.__bf = val

    def find_ideal_function(self, data):
        """Calculate the deviance between actual and predicted arrays"""

        try:
            for i in range(1, 50):
                col_name = f'y{i}'
                # Compute the RMSE and max error between
                # training function and each ideal function
                # The smallest MRE signals the closest match between functions
                rms_error = mean_squared_error(self.y, data.df[col_name], squared=False)
                if rms_error < self.rmse:
                    self.rmse = rms_error
                max_r = max_error(self.y, data.df[col_name])
                if max_r < self.max_dev:
                    self.max_dev = max_r
                    self.ideal_col = col_name
                    self.ideal_col_array = data.df[col_name]
        except ValueError:
            print("polynomial needs work")

    def match_ideal_functions(self, *args, **kwargs):
        """Match test data with its associated ideal function"""

        dev_list, funs_list = [], []
        idx = 0
        ideal_funs, train_df, models_ = args[0], args[1], args[2]
        map_train = kwargs.get('map_train', True)

        for row in self.df.itertuples(index=False):
            column, diff = "", 100000
            try:
                for c in ideal_funs.keys():
                    rmse = mean_squared_error([row.y], [ideal_funs.at[row.x, c]],
                                              squared=False)
                    if rmse < diff:
                        diff = rmse
                        column = c
            except KeyError as e:
                print(e)
                continue
            except ValueError as v:
                print(v)
                continue
            dev_list.append(diff)
            funs_list.append(column)

        # Append matched ideal functions to each test case
        self.df['delta_y'] = [float(i) for i in dev_list]
        self.df['ideal_fun'] = funs_list
        self.df.sort_values(by=['ideal_fun', 'x'], inplace=True)

        # Check if each ideal function is within the threshold of error
        # Turn value into NaN if not
        for row in self.df.itertuples(index=False):
            for i, j in models_.items():
                if row.ideal_fun == j.ideal_col:
                    _factor = np.sqrt(j.max_dev) + j.max_dev
                    if row.delta_y > _factor:
                        self.df.iat[idx, 3] = 'n/a'
            idx += 1

        if map_train:
            return self.map_test_with_train(train_df)
        else:
            return self.df

    def map_test_with_train(self, train_df):
        """Maps new test data with training function, to be plotted"""

        test_copy = self.df
        test_copy = test_copy.reset_index(drop=True)
        test_map = {}

        for row in test_copy.itertuples():
            # check training data to see which fun the ideal fun belongs to
            train_row = train_df[train_df['x'] == row.x]
            for i in range(1, 5):
                t = train_row[train_row[f'y{i}_if'].isin([row.ideal_fun])]
                y_col = f'y{i}'
                if not len(t.index) == 0:
                    nt = train_df
                    nt_ = nt[['x', y_col]]
                    new_train = nt_.set_index('x')
                    # match test(x, y) values with train(x, y) values
                    # to be plotted and tested
                    n = test_copy.loc[test_copy['ideal_fun'] == row.ideal_fun]
                    for r in n.itertuples():
                        ideal, train = row.ideal_fun, y_col
                        if train not in test_map.keys():
                            test_map[train] = {'ideal_fun': ideal, train: [], ideal: []}
                        else:
                            test_map[train][train].append((r.x, new_train.at[r.x, y_col]))
                            test_map[train][ideal].append((r.x, r.y))

        return self.compare_errors(test_map, train_df)

    def compare_errors(self, test_map, train_df):
        """Identify errors and plot processed data"""

        models = {}

        for k, v in test_map.items():
            ideal_ = test_map[k]['ideal_fun']
            test_arr = test_map[k][k]
            ideal_arr = test_map[k][ideal_]
            test_axes = list(starmap(lambda x, y: [x, y], test_arr))
            ideal_axes = list(starmap(lambda x, y: [x, y], ideal_arr))
            t_x = [ax[0] for ax in test_axes]
            t_y = [ax[1] for ax in test_axes]
            id_x = [ax[0] for ax in ideal_axes]
            id_y = [ax[1] for ax in ideal_axes]

            # Graph test vs ideal funs
            t_df = pd.DataFrame()
            t_df['x'] = t_x
            t_df['y'] = t_y
            plot_ = Graph(f'Ideal vs. Test', df=t_df)
            model = Model(id_x, id_y, ideal_[1:])

            # identify max error of training function
            c = f'{k}_max_err'
            tr_err = round(train_df[c][0], 6)

            # match ideal function with training function
            bf = f'{k}_best_fit'
            fit = train_df[bf][0]

            model.set_best_fit(fit)
            model.set_max_dev(tr_err)
            models[k] = model.__bf

            # plot graph
            plot_.plot_model(model, plt_type='test_vs_ideal', fit_model=model.__bf)

        return self.df, models
