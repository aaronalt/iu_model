"""
Aaron Althauser
IU International University of Applied Sciences
Data Science, M.Sc.
"""

from sqlalchemy import Table, MetaData, create_engine, Column, Float, String, inspect
from sqlalchemy.exc import InvalidRequestError
import pandas as pd
from numpy.polynomial import Polynomial as P
from scipy import stats
from xhtml2pdf import pisa
from model import Model
from visualize import Graph

meta = MetaData()
engine = create_engine("sqlite:///python_models.db", echo=True)


class TableNotCreatedException(Exception):
    """Called if SQLite Table not created"""
    pass


class RegressionException(Exception):
    """Called if type of regression not specified"""
    pass


class CSVError(Exception):
    """Called if csv file not found"""
    pass


class DBTable:
    """Creates a new sqlalchemy.Table object for an SQLite database"""

    def __init__(self, *names, **kwargs):
        self.name, = names
        self.conn = engine
        self.df = pd.DataFrame()
        self.table = Table()

        _create = kwargs.get('_create', True)
        to_db = kwargs.get('to_db', True)

        if _create:
            try:
                # Create graphs for SQLite
                if 'train' in self.name:
                    self.table = Table(
                        "training_data", meta,
                        Column("x", Float, primary_key=True),
                        Column("y1", Float),
                        Column("y2", Float),
                        Column("y3", Float),
                        Column("y4", Float))
                if 'test_' in self.name:
                    self.table = Table(
                        "test_data", meta,
                        Column("x", Float, primary_key=True),
                        Column("y", Float),
                        Column("delta y", Float),
                        Column("ideal function", Float))
                if 'ideal' in self.name:
                    self.table = Table(
                        "ideal_functions", meta,
                        Column("x", Float, primary_key=True),
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
            except InvalidRequestError:
                print('table already created')
                pass

            if to_db:
                self.csv_to_db()
            else:
                self.csv_to_df()

    def add_test_table(self, name):
        try:
            self.table = Table(
                name, meta,
                Column('a', String), Column('b', String), Column('c', String))
            self.table.create(self.conn, checkfirst=True)
            print('table added')
        except InvalidRequestError as ire:
            print(ire)

    def drop_test_table(self):
        self.table.drop(self.conn, checkfirst=True)
        inspector = inspect(self.conn)
        print('unittest' in inspector.get_table_names())

    def csv_to_db(self):
        """Reads data from CSV file, converts to Pandas Dataframe,
        inserts Dataframe into the database"""
        self.csv_to_df()
        self.df_to_db(pd.DataFrame())
        meta.create_all(engine)
        return True

    def csv_to_df(self):
        """Reads data from CSV file and converts into Pandas Dataframe"""
        if 'train' in self.name:
            self.df = pd.read_csv("./datasets/train.csv")
            return self.df
        elif 'test_' in self.name:
            self.df = pd.read_csv("./datasets/test.csv")
            return self.df
        elif 'ideal' in self.name:
            self.df = pd.read_csv("./datasets/ideal.csv")
            return self.df
        elif 'unittest' in self.name:
            self.df = pd.read_csv(f'./tests/{self.name}.csv')
        else:
            raise CSVError("csv file not found")

    def df_to_db(self, df):
        """Inserts Dataframe object into the database"""
        if df.__sizeof__() == 0:
            self.df.to_sql(self.name, con=engine, if_exists='replace')
            meta.create_all(engine)
        else:
            df.to_sql(self.name, con=engine, if_exists='replace')
            meta.create_all(engine)

    def df_to_html(self, df, name='table', output_dir='./tables'):
        """Writes HTML file from dataframe"""
        new_file = df.to_html(justify='center', index=False)
        new_file = new_file.replace('<table border="1" class="dataframe">',
                                    '<table style="border-top:2px solid black;'
                                    'border-bottom:1px solid black;font-family:Arial;" '
                                    'class="dataframe">')
        new_file = new_file.replace('<tr>',
                                    '<tr style="text-align:center;border:0;padding-top:2px;">')
        new_file = new_file.replace('<th>',
                                    '<th style="width:120px;'
                                    'padding-top:3px;margin-bottom:3px;font-size:11pt;">')
        new_file = new_file.replace('<tbody>',
                                    '<tbody style="padding-top:2px;font-size:10pt;">')

        # convert to .pdf
        with open(f'{name}.html', 'w') as f:
            self.html_to_pdf(new_file, f'{output_dir}/{name}.pdf')

    def html_to_pdf(self, source_html, output_filename):
        """Writes .pdf file from HTML"""
        with open(output_filename, "w+b") as f:
            pisa_status = pisa.CreatePDF(source_html, dest=f)
        return pisa_status.err


class Data(DBTable):
    """Inherits attributes from DBTable class.
    Returns best fit model as a Model() object."""

    def __init__(self, *names, **kwargs):
        super().__init__(*names, **kwargs)

    def is_empty(self):
        """Helper function to check if df of Data obj is empty"""
        if self.df.size == 0:
            return True
        else:
            return False

    def fit_model(self, *args, **kwargs):
        """Fit a regression model with training data function"""

        col = args[0]
        _ideal = args[1]
        r_type = args[2]

        order = kwargs.get('order', 1)
        subplot = kwargs.get('subplot', False)
        print_table = kwargs.get('print_table', False)
        table_name = kwargs.get('table_name', '')

        col_name = f'y{col}'
        subplot_array, _n, _rss, _rmse, _max_e, _var = [], [], [], [], [], []
        global model_df

        if r_type == "linear":
            x = self.df['x'].values
            lr = stats.linregress(self.df['x'], self.df[col_name])
            fun = lr.slope * x + lr.intercept
            model = Model(x, fun, col)
            try:
                # match ideal function if ideal Data object is passed
                if not _ideal.is_empty():
                    model.find_ideal_function(_ideal)
                    _rmse.append(model.rmse)
                    _max_e.append(model.max_dev)
                if print_table:
                    model_df = pd.DataFrame()
                    model_df['RMSE'] = [round(float(i), 5) for i in _rmse]
                    model_df['MRE'] = [round(float(i), 5) for i in _max_e]
                    if not table_name:
                        self.df_to_html(model_df, f'{col_name}_linear')
                    else:
                        out = table_name.rsplit('/', 1)
                        self.df_to_html(model_df, f'{out[1]}', output_dir=out[0])
                return model, model_df
            except AttributeError:
                # raised if _ideal df is empty
                raise Exception("ideal df empty or None")

        if r_type == "poly.fit":
            model = Model([], [], col)
            rss_max = 1000
            # Iterates through orders and returns fit with
            # minimum residual error, with weight=1/y
            for i in range(1, order + 1):
                weight = 1 / self.df[col_name]
                fn = P.fit(self.df['x'], self.df[col_name], i, full=True, w=weight)
                coeff, det = fn
                fn_x, fn_y = coeff.linspace(n=400)
                model = Model(fn_x, fn_y, col, rss=det[0], order=i)
                # match ideal function if ideal Data object is passed
                if not _ideal.is_empty():
                    model.find_ideal_function(_ideal)
                    _n.append(i)
                    _rss.append(model.rss)
                    _rmse.append(model.rmse)
                    _max_e.append(model.max_dev)
                if det[0] < rss_max:
                    subplot_array.append(model)
            if print_table:
                print(f'_n = {_n}')
                model_df = pd.DataFrame()
                model_df['Order'] = _n
                model_df['RSS'] = [i[0].round(5) for i in _rss]
                model_df['RMSE'] = [i.round(5) for i in _rmse]
                model_df['MRE'] = [i.round(5) for i in _max_e]
                self.df_to_html(model_df, f'{col_name}_order-{order}')
                print(f'from print_table:\n{model_df}')
                return model, model_df
            elif not subplot or len(subplot_array) <= 1:
                return model
            else:
                subplot_graph = Graph('Polynomial order and weighted NLR', df=self.df)
                subplot_graph.make_subplots(subplot_array)
                return model
        else:
            raise RegressionException("You must provide a valid type "
                                      "of regression via keyword arg.")
