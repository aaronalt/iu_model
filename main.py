"""
Aaron Althauser
IU International University of Applied Sciences
Data Science, M.Sc.
"""

from sqlalchemy import create_engine, MetaData
from database import Data
from model import Model
from visualize import Graph
import pandas as pd


engine = create_engine("sqlite:///python_models.db", echo=True)
meta = MetaData()
pd.set_option('display.float_format', lambda x: '%.5f' % x)


def main():
    """Main entry point of the program."""

    # Instatiate new Data object for training and ideal functions datasets
    train = Data("training_data", to_db=True)
    ideal = Data("ideal_functions", to_db=True)

    '''Plot training dataset as subplots'''
    train_graph = Graph("Training Data", train.csv_to_df())
    train_graph.make_subplots(train_graph.title)

    # Dictionary for different iterations' polynomial order
    _n = {
        'y1': [5, 21, 36],
        'y2': [5, 22, 36],
        'y4': [3, 9, 27]
    }

    '''
    Fit models to training data, map ideal function and plot
    '''

    # 1st iteration
    # Empty data objects to hold created models, processed data
    ideal_funs_dict = {'x': train.df['x']}
    train_master = train.csv_to_df()
    #train.df_to_html(train_master)
    models = {}

    # y1
    nl_p = train.fit_model(1, ideal, 'poly.fit', order=_n['y1'][0])
    models['y1'] = nl_p
    train_graph.plot_model(nl_p, 'np.Polynomial', 'best fit', with_rmse=True)
    ideal_funs_dict[nl_p.ideal_col] = nl_p.ideal_col_array
    train_master['y1_if'] = nl_p.ideal_col
    train_master['y1_max_err'] = nl_p.max_dev
    train_master['y1_best_fit'] = nl_p

    # y2
    nl_7 = train.fit_model(2, ideal, 'poly.fit', order=_n['y2'][0])
    models['y2'] = nl_7
    train_graph.plot_model(nl_7, 'poly.fit', 'best fit', with_rmse=True)
    ideal_funs_dict[nl_7.ideal_col] = nl_7.ideal_col_array
    train_master['y2_if'] = nl_7.ideal_col
    train_master['y2_max_err'] = nl_7.max_dev
    train_master['y2_best_fit'] = nl_7

    # y3
    lm_p = train.fit_model(3, ideal, 'poly.fit', 20, print_table=True)
    models['y3'] = lm_p
    train_graph.plot_model(lm_p, 'poly.fit', 'best fit', with_rmse=True)
    ideal_funs_dict[lm_p.ideal_col] = lm_p.ideal_col_array
    train_master['y3_if'] = lm_p.ideal_col
    train_master['y3_max_err'] = lm_p.max_dev
    train_master['y3_best_fit'] = lm_p

    # y4
    cm_p = train.fit_model(4, ideal, 'poly.fit', order=_n['y4'][0])
    models['y4'] = cm_p
    train_graph.plot_model(cm_p, 'poly.fit', 'best fit', with_rmse=True)
    ideal_funs_dict[cm_p.ideal_col] = cm_p.ideal_col_array
    train_master['y4_if'] = cm_p.ideal_col
    train_master['y4_max_err'] = cm_p.max_dev
    train_master['y4_best_fit'] = cm_p

    train_master['x'] = [round(i, 2) for i in train_master['x']]

    # Compare ideal functions to training data and choose best fit
    ideal_funs_df = pd.DataFrame(data=ideal_funs_dict)
    ideal_funs_df = ideal_funs_df.set_index('x')
    test = Data("test_data")
    test_df = test.csv_to_df()
    test_model = Model(test_df['x'], test_df['y'], 1, df=test_df)
    test_df_1, _tm1 = test_model.match_ideal_functions(ideal_funs_df, train_master, models)

    '''
    What did we discover from 1st iteration:
    - fn(y1) could use a lower-order polynomial for a more general fit without overfitting
      - test: lower order vs. 1st best fit
    - fn(y2): same as y1
    - fn(y3): needs linear and not polynomial function to reduce rss
      - test: new linear function vs. current best fit
    - fn(y4): best order is 3, but can rss be reduced? still too large 
    '''

    # 2nd iteration
    # New empty objects
    ideal_funs_dict = {'x': train.df['x']}
    train_master = train.csv_to_df()
    models_2 = {}

    # y1
    nl_p2 = train.fit_model(1, ideal, 'poly.fit', order=_n['y1'][1])
    models_2['y1'] = nl_p2
    train_graph.plot_model(nl_p2, 'np.Polynomial', 'best fit', with_rmse=True)
    ideal_funs_dict[nl_p.ideal_col] = nl_p2.ideal_col_array
    train_master['y1_if'] = nl_p2.ideal_col
    train_master['y1_max_err'] = nl_p2.max_dev
    train_master['y1_best_fit'] = nl_p2

    # y2
    nl_72 = train.fit_model(2, ideal, 'poly.fit', order=_n['y2'][1])
    models_2['y2'] = nl_72
    train_graph.plot_model(nl_72, 'poly.fit', 'best fit', with_rmse=True)
    ideal_funs_dict[nl_72.ideal_col] = nl_72.ideal_col_array
    train_master['y2_if'] = nl_72.ideal_col
    train_master['y2_max_err'] = nl_72.max_dev
    train_master['y2_best_fit'] = nl_72

    # y3
    lm_p2 = train.fit_model(3, ideal, 'linear')
    models_2['y3'] = lm_p2
    train_graph.plot_model(lm_p2, 'Linear', 'best fit', with_rmse=True)
    ideal_funs_dict[lm_p2.ideal_col] = lm_p2.ideal_col_array
    train_master['y3_if'] = lm_p2.ideal_col
    train_master['y3_max_err'] = lm_p2.max_dev
    train_master['y3_best_fit'] = lm_p2

    # y4
    cm_p2 = train.fit_model(4, ideal, 'poly.fit', order=_n['y4'][1])
    models_2['y4'] = cm_p2
    train_graph.plot_model(cm_p2, 'poly.fit', 'best fit', with_rmse=True)
    ideal_funs_dict[cm_p2.ideal_col] = cm_p2.ideal_col_array
    train_master['y4_if'] = cm_p2.ideal_col
    train_master['y4_max_err'] = cm_p2.max_dev
    train_master['y4_best_fit'] = cm_p2

    # Compare functions
    ideal_funs_df = pd.DataFrame(data=ideal_funs_dict)
    ideal_funs_df = ideal_funs_df.set_index('x')
    test_model = Model(test_df['x'], test_df['y'], 1, df=test_df)
    test_df_2, _tm2 = test_model.match_ideal_functions(ideal_funs_df, train_master, models_2)

    # 3rd iteration
    # New empty objects
    ideal_funs_dict = {'x': train.df['x']}
    train_master = train.csv_to_df()
    models_3 = {}

    # y1
    nl_p3 = train.fit_model(1, ideal, 'poly.fit', order=_n['y1'][2], print_table=True)
    models_3['y1'] = nl_p3
    train_graph.plot_model(nl_p3, 'np.Polynomial', 'best fit', with_rmse=True)
    ideal_funs_dict[nl_p3.ideal_col] = nl_p3.ideal_col_array
    train_master['y1_if'] = nl_p3.ideal_col
    train_master['y1_max_err'] = nl_p3.max_dev
    train_master['y1_best_fit'] = nl_p3

    # y2
    nl_73 = train.fit_model(2, ideal, 'poly.fit', order=_n['y2'][2], print_table=True)
    models_3['y2'] = nl_73
    train_graph.plot_model(nl_73, 'poly.fit', 'best fit', with_rmse=True)
    ideal_funs_dict[nl_73.ideal_col] = nl_73.ideal_col_array
    train_master['y2_if'] = nl_73.ideal_col
    train_master['y2_max_err'] = nl_73.max_dev
    train_master['y2_best_fit'] = nl_73

    # y3
    lm_p3 = train.fit_model(3, ideal, 'linear')
    models_3['y3'] = lm_p3
    train_graph.plot_model(lm_p3, 'Linear', 'best fit', with_rmse=True)
    ideal_funs_dict[lm_p3.ideal_col] = lm_p3.ideal_col_array
    train_master['y3_if'] = lm_p3.ideal_col
    train_master['y3_max_err'] = lm_p3.max_dev
    train_master['y3_best_fit'] = lm_p3

    # y4
    cm_p3 = train.fit_model(4, ideal, 'poly.fit', order=_n['y4'][2], print_table=True)
    models_3['y4'] = cm_p3
    train_graph.plot_model(cm_p3, 'poly.fit', 'best fit', with_rmse=True)
    ideal_funs_dict[cm_p3.ideal_col] = cm_p3.ideal_col_array
    train_master['y4_if'] = cm_p3.ideal_col
    train_master['y4_max_err'] = cm_p3.max_dev
    train_master['y4_best_fit'] = cm_p3

    # Compare functions
    ideal_funs_df = pd.DataFrame(data=ideal_funs_dict)
    ideal_funs_df = ideal_funs_df.set_index('x')
    test_model = Model(test_df['x'], test_df['y'], 1, df=test_df)
    test_df_3, _tm3 = test_model.match_ideal_functions(ideal_funs_df, train_master, models_3)

    # Plot comparisons between polynomial orders for each training function
    train_graph.make_subplots('Model Comparison', model_1=models, model_2=models_2, model_3=models_3)

    # Insert test df into database
    # test.df_to_db(new_test_df)

    # todo: unit tests


if __name__ == "__main__":
    main()
