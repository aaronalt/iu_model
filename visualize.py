"""
Aaron Althauser
IU International University of Applied Sciences
Data Science, M.Sc.
"""


from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sb
import numpy as np


class Graph:
    """Graph object to plot models and data"""

    def __init__(self, title, df=pd.DataFrame()):
        self.title = title
        self.plt = plt
        self.data = df
        self.x = self.data['x']

    def make_subplots(self, title=str(), model_1=None, model_2=None, model_3=None):
        """Plot multiple subplots at once"""

        sb.set_theme()
        idx = 0

        if 'train' in title.lower():
            fig, axs = self.plt.subplots(4, 1, sharex=True)
            for i in self.data.keys():
                if i != "x":
                    temp = self.data[i]
                    y = [float(i) for i in temp.to_list()]
                    y_label = f'y{int(idx) + 1}'
                    axs[idx].plot(self.x, y, color='green', alpha=0.8, marker='o', linewidth=0, markersize=2)
                    axs[idx].set(ylabel=y_label)
                    if idx == 0:
                        axs[idx].set(title=self.title)
                    idx += 1

            fig.align_ylabels()
            axs[idx - 1].set(xlabel='x')
            self.plt.tight_layout()
            self.plt.savefig(f'./graphs/{title}.pdf', bbox_inches='tight')
            self.plt.show()

        if 'model' in title.lower():

            for fn, _m in model_1.items():

                box = {'facecolor': 'white', 'alpha': 0.8, 'edgecolor': 'black'}
                fig, axs = self.plt.subplots(1, 3, sharey=True, figsize=(15, 5))

                try:
                    _m2, _m3 = model_2[fn], model_3[fn]
                    _m_rmse, _m_rss, _m_max = round(_m.rmse, 4), round(_m.rss[0], 4), round(_m.max_dev, 4)
                    _m2_rmse, _m2_rss, _m2_max = round(_m2.rmse, 4), round(_m2.rss[0], 4), round(_m2.max_dev, 4)
                    _m3_rmse, _m3_rss, _m3_max = round(_m3.rmse, 4), round(_m3.rss[0], 4), round(_m3.max_dev, 4)
                    factor = np.sqrt(_m.max_dev) + _m.max_dev

                    axs[0].fill_between(_m.x, _m.y - factor, _m.y + factor, alpha=0.5)
                    axs[0].plot(_m.x, _m.y)
                    axs[0].scatter(self.data['x'], self.data[fn])
                    axs[0].annotate(f'n = {_m.order}\n'
                                    f'RMSE = {_m_rmse}\n'
                                    f'RSS = {_m_rss}\n'
                                    f'MRE = {_m_max}',
                                    xy=(2, 2), xycoords='axes points', bbox=box, fontsize=10)
                    axs[0].set_ylabel('y')

                    axs[1].fill_between(_m2.x, _m2.y - factor, _m2.y + factor, alpha=0.5, label='error')
                    axs[1].plot(_m2.x, _m2.y, label='best fit')
                    axs[1].scatter(self.data['x'], self.data[fn], label='training data')
                    axs[1].annotate(f'n = {_m2.order}\n'
                                    f'RMSE = {_m2_rmse}\n'
                                    f'RSS = {_m2_rss}\n'
                                    f'MRE = {_m2_max}',
                                    xy=(2, 2), xycoords='axes points', bbox=box, fontsize=10)
                    axs[1].legend()
                    axs[1].set_xlabel('x')
                    axs[1].set(title=f'{title}, {fn}')

                    axs[2].fill_between(_m3.x, _m3.y - factor, _m3.y + factor, alpha=0.5)
                    axs[2].plot(_m3.x, _m3.y)
                    axs[2].scatter(self.data['x'], self.data[fn])
                    axs[2].annotate(f'n = {_m3.order}\n'
                                    f'RMSE = {_m3_rmse}\n'
                                    f'RSS = {_m3_rss}\n'
                                    f'MRE = {_m3_max}',
                                    xy=(2, 2), xycoords='axes points', bbox=box, fontsize=10)

                    fig.subplots_adjust(wspace=0.1)
                    self.plt.savefig(f'./graphs/{fn}_{title}.pdf', bbox_inches='tight')
                    self.plt.show()
                except TypeError:
                    '''Occurs when type is linear'''
                    continue

    def plot_model(self, model_, descr_title='', plt_type=None, error=False, with_rmse=False, fit_model=None):
        """Plot models using matplotlib"""

        sb.set_theme()
        col_name = f'y{model_.col}'

        if fit_model:
            self.plt.title(f'{self.title}, {col_name} [n = {fit_model.order}]')
        else:
            self.plt.title(f'{self.title}, {col_name} [n = {model_.order}]')

        if plt_type == 'best fit':
            y = self.data[col_name]
            self.plt.scatter(self.x, y, label="data", alpha=0.4, color='green')

            if error:
                self.plt.fill_between(self.x, model_.y - model_.rmse, model_.y + model_.rmse,
                                      label='+/- RMSE', alpha=0.2)
            if with_rmse:
                self.plt.plot(self.x, model_.ideal_col_array, "o", linewidth=0, label="ideal", color='blue')

            self.plt.plot(model_.x, model_.y, "-", label="best fit", linewidth=2, color='red')
            self.plt.xlabel('x')
            self.plt.ylabel('y')
            self.plt.legend()
            self.plt.savefig(f'./graphs/{col_name}_order-{model_.order}_bestfit.pdf', bbox_inches='tight')
            self.plt.show()

        if plt_type == 'test_vs_ideal':
            factor = np.sqrt(fit_model.max_dev) + fit_model.max_dev
            self.plt.fill_between(fit_model.x, fit_model.y - factor, fit_model.y + factor, alpha=0.4,
                                  label="+/- max dev")
            self.plt.plot(fit_model.x, fit_model.y, "--", color="green", linewidth=2, markersize=0,
                          label="best fit")
            self.plt.plot(self.x, self.data['y'], "o", color="blue", linewidth=0, markersize=7, label="ideal")
            self.plt.plot(model_.x, model_.y, marker="^", color="red", linewidth=0, markersize=6,
                          label="test")
            self.plt.xlabel('x')
            self.plt.ylabel('y')
            self.plt.legend()
            self.plt.savefig(f'./graphs/y{fit_model.col}_order-{fit_model.order}_test-vs-ideal.pdf',
                             bbox_inches='tight')
            self.plt.show()
