import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from collections import OrderedDict
import pandas as pd
from .evaluator_helpers import Categories, Sub_categories, Metrics


class Table(object):
    """docstring for Table"""
    def __init__(self, arg=None):
        super(Table, self).__init__()
        self.entries = {}
        self.sub_entries = {}
        self.arg = arg
        self.results = {}
        self.sub_results = {}


    def add_entry(self, name, results):
        final_results = []
        sub_final_results = []
        ## Overall metrics ADE, FDE, Topk_ade, Topk_fde, NLL
        table_metrics = Metrics(*([0]*6))
        ## Metrics for the 2 types of trajectories and 4 types of interactions
        table_categories = Categories(*[Metrics(*([0]*3)) for i in range(1,3)])
        table_sub_categories = Sub_categories(*[Metrics(*([0]*3)) for i in range(1,8)])

        for dataset, (metrics, categories, sub_categories) in results.items():
            ## Overall
            table_metrics += metrics
            
            ## Main Types
            table_categories.highD_scenes += categories.highD_scenes
            table_categories.inD_scenes += categories.inD_scenes

            ## Sub Types
            table_sub_categories.static_highD += sub_categories.static_highD
            table_sub_categories.lane_highD += sub_categories.lane_highD
            table_sub_categories.linear_highD += sub_categories.linear_highD
            table_sub_categories.others_highD += sub_categories.others_highD

            table_sub_categories.static_inD += sub_categories.static_inD
            table_sub_categories.linear_inD += sub_categories.linear_inD
            table_sub_categories.others_inD += sub_categories.others_inD

        final_results += table_categories.highD_scenes.avg_vals_to_list()
        final_results += table_categories.inD_scenes.avg_vals_to_list()

        final_results += table_metrics.avg_vals_to_list()

        sub_final_results += table_sub_categories.static_highD.avg_vals_to_list()
        sub_final_results += table_sub_categories.lane_highD.avg_vals_to_list()
        sub_final_results += table_sub_categories.linear_highD.avg_vals_to_list()
        sub_final_results += table_sub_categories.others_highD.avg_vals_to_list()

        sub_final_results += table_sub_categories.static_inD.avg_vals_to_list()
        sub_final_results += table_sub_categories.linear_inD.avg_vals_to_list()
        sub_final_results += table_sub_categories.others_inD.avg_vals_to_list()

        self.results[name] = final_results
        self.sub_results[name] = sub_final_results
        return final_results, sub_final_results

    def add_result(self, name, final_results, sub_final_results):
        self.results[name] = final_results
        self.sub_results[name] = sub_final_results

    def render_mpl_table(self, data, col_width=3.0, row_height=0.625, font_size=14,
                             header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                             bbox=[0, 0, 1, 1], header_columns=0,
                             ax=None, **kwargs):
        if ax is None:
            size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
            fig, ax = plt.subplots(figsize=size)
            ax.axis('off')

        mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, cellLoc='center', **kwargs)

        for (row, col), cell in mpl_table.get_celld().items():
            if (row == 0) or (col == 1) or (col == 0):
                cell.set_text_props(fontproperties=FontProperties(weight='bold'))

            # If the cell's value is numeric and is the minimum in its row, bold its border
            if row > 0 and col > 1:  # Assuming the first 2 columns are non-numeric
                try:
                    cell_value = float(cell.get_text().get_text())
                    if cell_value == min(data.iloc[row - 1, 2:].astype(float)):
                        cell.set_edgecolor('black')
                        cell.set_linewidth(2.0)
                except ValueError:
                    pass

        mpl_table.auto_set_font_size(False)
        mpl_table.set_fontsize(font_size)

        return ax
    # def highlight_max_min(s, max_val, min_val):
    #     if s == max_val:
    #         return 'background-color: yellow'
    #     elif s == min_val:
    #         return 'background-color: cyan'
    #     return ''
    #
    # def render_mpl_table(self, data, col_width=3.0, row_height=0.625, font_size=12,
    #                      header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
    #                      bbox=[0, 0, 1, 1], header_columns=0,
    #                      ax=None, **kwargs):
    #     if ax is None:
    #         size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
    #         fig, ax = plt.subplots(figsize=size)
    #         ax.axis('off')
    #     mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)
    #
    #     data['ADE'] = pd.to_numeric(data['ADE'], errors='coerce')
    #     data['FDE'] = pd.to_numeric(data['FDE'], errors='coerce')
    #
    #     # Set max and min values for ADE and FDE
    #     if 'Type' in data.columns and 'Sub-Type' in data.columns:
    #         ade_max_min = data.groupby(['Type', 'Sub-Type'])['ADE'].agg(['idxmax', 'idxmin']).values.flatten()
    #         fde_max_min = data.groupby(['Type', 'Sub-Type'])['FDE'].agg(['idxmax', 'idxmin']).values.flatten()
    #         highlight_indices = np.unique(np.concatenate([ade_max_min, fde_max_min]))
    #     else:
    #
    #
    #         ade_max, ade_min = data['ADE'].idxmax(), data['ADE'].idxmin()
    #         fde_max, fde_min = data['FDE'].idxmax(), data['FDE'].idxmin()
    #         highlight_indices = [ade_max, ade_min, fde_max, fde_min]
    #
    #     for i, (index, row) in enumerate(data.iterrows()):
    #         # Highlight ADE and FDE
    #         if i in highlight_indices:
    #             for j, val in enumerate(row):
    #                 cell = mpl_table[i + 1, j]
    #                 if j == list(data.columns).index('ADE'):
    #                     cell.set_facecolor('yellow' if val == data['ADE'].max() else 'cyan')
    #                 elif j == list(data.columns).index('FDE'):
    #                     cell.set_facecolor('yellow' if val == data['FDE'].max() else 'cyan')
    #                 cell.set_edgecolor('black')
    #
    #     mpl_table.auto_set_font_size(False)
    #     mpl_table.set_fontsize(font_size)
    #
    #     for k, cell in mpl_table._cells.items():
    #         cell.set_edgecolor(edge_color)
    #         if k[0] == 0 or k[1] < header_columns:
    #             cell.set_text_props(weight='bold', color='w')
    #             cell.set_facecolor(header_color)
    #         else:
    #             cell.set_facecolor(row_colors[k[0] % len(row_colors)])
    #     return ax

    def print_table(self):
        fig = plt.figure(figsize=(20, 20))
    # ------------------------------------------ TABLES -------------------------------------------
        # Overall Table #
        ax1 = fig.add_subplot(311)
        ax1.axis('tight')
        ax1.axis('off')

        df = pd.DataFrame(columns=['', 'Model', 'No.', 'ADE', 'FDE'])
        it = 0
        len_name = 13
        for key in self.results:
            df.loc[it] = ['Overall'] + [key[:len_name]] + [self.results[key][index].__format__('.2f') for index in range(6, 9)]
            it += 1

        ax1 = self.render_mpl_table(df, header_columns=0, col_width=2.0, bbox=[0, 0.9, 1, 0.1*len(self.results)], ax=ax1)

        ax2 = fig.add_subplot(312)
        ax2.axis('tight')
        ax2.axis('off')
        # Overall Table #

        df = pd.DataFrame(columns=['Type', 'Sub-Type', 'Model', 'No.', 'ADE', 'FDE'])

        type_list = [['highD', ''], ['highD', 'Static'], ['highD', 'Lane'], ['highD', 'Linear'], ['highD', 'Others'],
                     ['inD', ''], ['inD', 'Static'], ['inD', 'Linear'], ['inD', 'Others']]
        it = 0

        ##Type I highD
        for key in self.results:
            df.loc[it] = type_list[0] + [key[:len_name]] + [self.results[key][index].__format__('.2f') for index in range(3)]
            it += 1

        df.loc[it] = ['Type', 'Sub-Type', 'Model', 'No.', 'ADE', 'FDE']
        it += 1

        ##Type I highD: Statics
        for key in self.results:
            df.loc[it] = type_list[1] + [key[:len_name]] + [self.sub_results[key][index].__format__('.2f') for index in range(3)]
            it += 1

        df.loc[it] = ['Type', 'Sub-Type', 'Model', 'No.', 'ADE', 'FDE']
        it += 1

        ##Type I highD: Lane
        for key in self.results:
            df.loc[it] = type_list[2] + [key[:len_name]] + [self.sub_results[key][index].__format__('.2f') for index in range(3, 6)]
            it += 1

        df.loc[it] = ['Type', 'Sub-Type', 'Model', 'No.', 'ADE', 'FDE']
        it += 1

        ##Type I highD: Linear
        for key in self.results:
            df.loc[it] = type_list[3] + [key[:len_name]] + [self.sub_results[key][index].__format__('.2f') for index in range(6, 9)]
            it += 1

        df.loc[it] = ['Type', 'Sub-Type', 'Model', 'No.', 'ADE', 'FDE']
        it += 1

        ##Type I highD: Others
        for key in self.results:
            df.loc[it] = type_list[4] + [key[:len_name]] + [self.sub_results[key][index].__format__('.2f') for index in range(9, 12)]
            it += 1

        df.loc[it] = ['Type', 'Sub-Type', 'Model', 'No.', 'ADE', 'FDE']
        it += 1

        ##Type II inD
        for key in self.results:
            df.loc[it] = type_list[5] + [key[:len_name]] + [self.results[key][index].__format__('.2f') for index in range(3, 6)]
            it += 1

        df.loc[it] = ['Type', 'Sub-Type', 'Model', 'No.', 'ADE', 'FDE']
        it += 1

        ##Type II inD: Statics
        for key in self.results:
            df.loc[it] = type_list[6] + [key[:len_name]] + [self.sub_results[key][index].__format__('.2f') for index in range(12, 15)]
            it += 1

        df.loc[it] = ['Type', 'Sub-Type', 'Model', 'No.', 'ADE', 'FDE']
        it += 1

        ##Type II inD: Linear
        for key in self.results:
            df.loc[it] = type_list[7] + [key[:len_name]] + [self.sub_results[key][index].__format__('.2f') for index in range(15, 18)]
            it += 1

        df.loc[it] = ['Type', 'Sub-Type', 'Model', 'No.', 'ADE', 'FDE']
        it += 1

        ##Type II inD: Others
        for key in self.results:
            df.loc[it] = type_list[8] + [key[:len_name]] + [self.sub_results[key][index].__format__('.2f') for index in range(18,21)]
            it += 1


        ax2 = self.render_mpl_table(df, header_columns=0, col_width=2.0, bbox=[0, -1.6, 1, 0.6*len(self.results)], ax=ax2)
        plt.subplots_adjust(hspace=0.5)  # 调整该值以更改子图之间的间距

        fig.savefig('Results.png')

    # def print_table(self):
    #     fig = plt.figure(figsize=(20, 20))
    # # ------------------------------------------ TABLES -------------------------------------------
    #     # Overall Table #
    #     ax1 = fig.add_subplot(311)
    #     ax1.axis('tight')
    #     ax1.axis('off')
    #
    #     df = pd.DataFrame(columns=['', 'Model', 'No.', 'ADE', 'FDE', 'Top3 ADE', 'Top3 FDE', 'NLL'])
    #     it = 0
    #     len_name = 13
    #     for key in self.results:
    #         df.loc[it] = ['Overall'] + [key[:len_name]] + [self.results[key][index].__format__('.2f') for index in range(12, 18)]
    #         it += 1
    #
    #     ax1 = self.render_mpl_table(df, header_columns=0, col_width=2.0, bbox=[0, 0.9, 1, 0.1*len(self.results)], ax=ax1)
    #
    #     ax2 = fig.add_subplot(312)
    #     ax2.axis('tight')
    #     ax2.axis('off')
    #     # Overall Table #
    #
    #     df = pd.DataFrame(columns=['Type', 'Sub-Type', 'Model', 'No.', 'ADE', 'FDE', 'Top3 ADE', 'Top3 FDE', 'NLL'])
    #
    #     type_list = [['highD', ''], ['highD', 'Static'], ['highD', 'Lane'], ['highD', 'Linear'], ['highD', 'Others'],
    #                  ['inD', ''], ['inD', 'Static'], ['inD', 'Linear'], ['inD', 'Others']]
    #     it = 0
    #
    #     ##Type I highD
    #     for key in self.results:
    #         df.loc[it] = type_list[0] + [key[:len_name]] + [self.results[key][index].__format__('.2f') for index in range(6)]
    #         it += 1
    #
    #     df.loc[it] = ['Type', 'Sub-Type', 'Model', 'No.', 'ADE', 'FDE', 'Top3 ADE', 'Top3 FDE', 'NLL']
    #     it += 1
    #
    #     ##Type I highD: Statics
    #     for key in self.results:
    #         df.loc[it] = type_list[1] + [key[:len_name]] + [self.sub_results[key][index].__format__('.2f') for index in range(6)]
    #         it += 1
    #
    #     df.loc[it] = ['Type', 'Sub-Type', 'Model', 'No.', 'ADE', 'FDE', 'Top3 ADE', 'Top3 FDE', 'NLL']
    #     it += 1
    #
    #     ##Type I highD: Lane
    #     for key in self.results:
    #         df.loc[it] = type_list[2] + [key[:len_name]] + [self.sub_results[key][index].__format__('.2f') for index in range(6, 12)]
    #         it += 1
    #
    #     df.loc[it] = ['Type', 'Sub-Type', 'Model', 'No.', 'ADE', 'FDE', 'Top3 ADE', 'Top3 FDE', 'NLL']
    #     it += 1
    #
    #     ##Type I highD: Linear
    #     for key in self.results:
    #         df.loc[it] = type_list[3] + [key[:len_name]] + [self.sub_results[key][index].__format__('.2f') for index in range(12, 18)]
    #         it += 1
    #
    #     df.loc[it] = ['Type', 'Sub-Type', 'Model', 'No.', 'ADE', 'FDE', 'Top3 ADE', 'Top3 FDE', 'NLL']
    #     it += 1
    #
    #     ##Type I highD: Others
    #     for key in self.results:
    #         df.loc[it] = type_list[4] + [key[:len_name]] + [self.sub_results[key][index].__format__('.2f') for index in range(18, 24)]
    #         it += 1
    #
    #     df.loc[it] = ['Type', 'Sub-Type', 'Model', 'No.', 'ADE', 'FDE', 'Top3 ADE', 'Top3 FDE', 'NLL']
    #     it += 1
    #
    #     ##Type II inD
    #     for key in self.results:
    #         df.loc[it] = type_list[5] + [key[:len_name]] + [self.results[key][index].__format__('.2f') for index in range(6, 12)]
    #         it += 1
    #
    #     df.loc[it] = ['Type', 'Sub-Type', 'Model', 'No.', 'ADE', 'FDE', 'Top3 ADE', 'Top3 FDE', 'NLL']
    #     it += 1
    #
    #     ##Type II inD: Statics
    #     for key in self.results:
    #         df.loc[it] = type_list[6] + [key[:len_name]] + [self.sub_results[key][index].__format__('.2f') for index in range(24, 30)]
    #         it += 1
    #
    #     df.loc[it] = ['Type', 'Sub-Type', 'Model', 'No.', 'ADE', 'FDE', 'Top3 ADE', 'Top3 FDE', 'NLL']
    #     it += 1
    #
    #     ##Type II inD: Linear
    #     for key in self.results:
    #         df.loc[it] = type_list[7] + [key[:len_name]] + [self.sub_results[key][index].__format__('.2f') for index in range(30, 36)]
    #         it += 1
    #
    #     df.loc[it] = ['Type', 'Sub-Type', 'Model', 'No.', 'ADE', 'FDE', 'Top3 ADE', 'Top3 FDE', 'NLL']
    #     it += 1
    #
    #     ##Type II inD: Others
    #     for key in self.results:
    #         df.loc[it] = type_list[8] + [key[:len_name]] + [self.sub_results[key][index].__format__('.2f') for index in range(36,42)]
    #         it += 1
    #
    #
    #     ax2 = self.render_mpl_table(df, header_columns=0, col_width=2.0, bbox=[0, -1.6, 1, 0.6*len(self.results)], ax=ax2)
    #     plt.subplots_adjust(hspace=0.5)  # 调整该值以更改子图之间的间距
    #
    #     fig.savefig('Results.png')
    #
    #
    