"""
Aaron Althauser
IU International University of Applied Sciences
Data Science, M.Sc.
"""


from sqlalchemy import Table, MetaData, create_engine, Column, Float
import pandas as pd
from numpy.polynomial import Polynomial as P
from scipy import stats
from xhtml2pdf import pisa
from model import Model
from visualize import Graph

engine = create_engine("sqlite:///python_models.db", echo=True)
meta = MetaData()


class TableNotCreatedException(Exception):
    """Called if SQLite Table not created"""
    pass


class DBTable:
    """
    Creates a new sqlalchemy.Table object for an SQLite database upon initialization.
    """

    def __init__(self, name):
        self.name = name
        self.conn = engine
        self.df = pd.DataFrame()
        self.table = Table()

        # Create graphs for SQLite
        if self.name == "training_data":
            self.table = Table(
                "training_data", meta,
                Column("x", Float),
                Column("y1", Float),
                Column("y2", Float),
                Column("y3", Float),
                Column("y4", Float))
        if self.name == "test_data":
            self.table = Table(
                "test_data", meta,
                Column("x", Float),
                Column("y", Float),
                Column("delta y", Float),
                Column("ideal function", Float))
        if self.name == "ideal_functions":
            self.table = Table(
                "ideal_functions", meta,
                Column("x", Float),
                Column("y1", Float), Column("y2", Float), Column("y3", Float), Column("y4", Float),
                Column("y5", Float), Column("y6", Float), Column("y7", Float), Column("y8", Float),
                Column("y9", Float), Column("y10", Float), Column("y11", Float), Column("y12", Float),
                Column("y13", Float), Column("y14", Float), Column("y15", Float), Column("y16", Float),
                Column("y17", Float), Column("y18", Float), Column("y19", Float), Column("y20", Float),
                Column("y21", Float), Column("y22", Float), Column("y23", Float), Column("y24", Float),
                Column("y25", Float), Column("y26", Float), Column("y27", Float), Column("y28", Float),
                Column("y29", Float), Column("y30", Float), Column("y31", Float), Column("y32", Float),
                Column("y33", Float), Column("y34", Float), Column("y35", Float), Column("y36", Float),
                Column("y37", Float), Column("y38", Float), Column("y39", Float), Column("y40", Float),
                Column("y41", Float), Column("y42", Float), Column("y43", Float), Column("y44", Float),
                Column("y45", Float), Column("y46", Float), Column("y47", Float), Column("y48", Float),
                Column("y49", Float), Column("y50", Float))


    def csv_to_db(self):
        """
        Reads data from CSV file, converts to Pandas Dataframe, inserts Dataframe into the database.
        """
        self.csv_to_df()
        self.df_to_db(pd.DataFrame())
        meta.create_all(engine)

    def csv_to_df(self):
        """
        Reads data from CSV file and converts into Pandas Dataframe.
        """
        if self.name == "training_data":
            self.df = pd.read_csv("./datasets/train.csv")
        if self.name == "test_data":
            self.df = pd.read_csv("./datasets/test.csv")
        if self.name == "ideal_functions":
            self.df = pd.read_csv("./datasets/ideal.csv")
        return self.df

    def df_to_db(self, df):
        """
        Inserts Dataframe object into the database.
        """
        try:
            if df.__sizeof__() == 0:
                self.df.to_sql(self.name, con=engine, if_exists='replace')
            else:
                df.to_sql(self.name, con=engine, if_exists='replace')
                meta.create_all(engine)
        except ValueError:
            # throw exception here
            print("table already exists")

    def df_to_html(self, df, name='table'):
        """"""
        # df = df.style.set_properties(**{'text-align': 'center'}).render()
        new_file = df.to_html(justify='center', index=False)
        new_file = new_file.replace('<table border="1" class="dataframe">',
                                    '<table style="border-top:2px solid black;'
                                    'border-bottom:1px solid black;font-family:Arial;" '
                                    'class="dataframe">')
        new_file = new_file.replace('<tr>', '<tr style="text-align:center;border:0;padding-top:2px;">')
        new_file = new_file.replace('<th>', '<th style="width:120px;'
                                            'padding-top:3px;margin-bottom:3px;font-size:11pt;">')
        new_file = new_file.replace('<tbody>', '<tbody style="padding-top:2px;font-size:10pt;">')

        with open(f'{name}.html', 'w') as f:
            self.html_to_pdf(new_file, f'./tables/{name}.pdf')  # convert to .pdf
            del f

    def html_to_pdf(self, source_html, output_filename):
        """"""
        with open(output_filename, "w+b") as f:
            pisa_status = pisa.CreatePDF(
                source_html,
                dest=f)
        return pisa_status.err


class Data(DBTable):
    """
    Class for mathematical functions and model creation.
    Inherits attributes from DBTable class. Returns best fit model as Model object.
    """

    def __init__(self, name):
        super().__init__(name)

    def fit_model(self, col, _ideal, r_type=None, order=1, subplot=False, print_table=True):
        """
        Fit a regression model with training data function.
        :param _ideal:
        :param r_type: Keyword arg., regression type ['linear', 'curve', 'poly', 'lorentz].
        :param col: Column of dataframe to apply.
        :return: Model object
        """
        col_name = f'y{col}'
        x = self.df['x'].values
        subplot_array = []
        _n = []
        _rss = []
        _rmse = []
        _max_e = []
        _var = []

        if r_type == "linear":
            lr = stats.linregress(self.df['x'], self.df[col_name])
            fun = lr.slope * x + lr.intercept
            model = Model(x, fun, col)
            if _ideal:  # match ideal function if ideal Data object is passed
                model.find_ideal_function(_ideal)
            _rmse.append(model.rmse)
            _max_e.append(model.max_dev)
            if print_table:
                model_df = pd.DataFrame()
                model_df['RMSE'] = [i.round(5) for i in _rmse]
                model_df['MRE'] = [i.round(5) for i in _max_e]
                self.df_to_html(model_df, f'{col_name}_linear')  # added order to create unique filename
                return model
            return model

        if r_type == "poly.fit":
            model = Model([], [], col)
            rss_max = 1000
            # Iterates through orders and returns fit with minimum residual error, with weight=1/y
            # todo: experiment with different weights, programatically (for paper, what to do next)
            for i in range(1, order+1):
                weight = 1 / self.df[col_name]
                fn = P.fit(self.df['x'], self.df[col_name], i, full=True, w=weight)
                coeff, det = fn
                fn_x, fn_y = coeff.linspace(n=400)
                model = Model(fn_x, fn_y, col, rss=det[0], order=i)
                if _ideal:  # match ideal function if ideal Data object is passed
                    model.find_ideal_function(_ideal)
                _n.append(i)
                _rss.append(model.rss)
                _rmse.append(model.rmse)
                _max_e.append(model.max_dev)
                _var.append(model.var)
                if det[0] < rss_max:
                    subplot_array.append(model)
            if print_table:
                model_df = pd.DataFrame()
                model_df['Order'] = _n
                model_df['RSS'] = [i[0].round(5) for i in _rss]
                model_df['RMSE'] = [i.round(5) for i in _rmse]
                model_df['MRE'] = [i.round(5) for i in _max_e]
                self.df_to_html(model_df, f'{col_name}_order-{order}')
                return model
            elif not subplot or len(subplot_array) <= 1:
                return model
            else:
                subplot_graph = Graph('Polynomial order and weighted NLR', self.df)
                subplot_graph.make_subplots(subplot_array)
                return model
        else:
            raise Exception("You must provide a type of regression via keyword arg.")
