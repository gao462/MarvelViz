import re
import numpy as np
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.palettes import *
from bokeh.models import Range1d, LabelSet
from bokeh.models.widgets import Div, CheckboxGroup, Slider, RangeSlider, Panel, Tabs
from bokeh.models import Button, TextInput
from bokeh.models.glyphs import ImageURL
from bokeh.layouts import row, column
from decimal import Decimal
from fuzzywuzzy import fuzz
from google_images_download import google_images_download
from PIL import Image
from .plot import GraphViz

class GraphWidget(object):
    r"""Graph Interaction Widget"""
    # constants
    CBOX_W = 150
    CBOX_H = 50
    CBOX_H_TITLE = 10

    SLDR_W = 300
    SLDR_H = 30
    
    RNG_SLDR_W = 300
    RNG_SLDR_H = 30

    PIE_W = 300
    PIE_H = 250

    BAR_W = 300
    BAR_H = 250

    def __init__(self, viz):
        r"""Initialize the class

        Args
        ----
        viz : GraphViz
            Graph visualization.

        """
        # get plot sharing data
        self.viz = viz
        self.node_renderer = self.viz.renderer.node_renderer
        self.edge_renderer = self.viz.renderer.edge_renderer        
        self.node_source = self.node_renderer.data_source
        self.edge_source = self.edge_renderer.data_source
        self.node_data = self.node_source.to_df()
        self.edge_data = self.edge_source.to_df()
        self.adj_source = self.viz.adj_source

        # deploy selection's trigger
        self.select_lock = 0
        self.node_source.selected.on_change('indices', lambda attr, old, new: self.select())

        # deploy checkbox and its trigger
        self.checkbox_()

        # deploy slider and its trigger
        self.slider_()

        # deploy slider and its trigger
        self.range_slider_()

        # deploy pie and bar chart
        self.bin_()
        self.pie_()
        self.bar_()

        # configure layout
        layout_widget = column(
            children=[self.layout_checkbox, self.layout_slider, self.layout_range_slider])
        layout_graph = row(children=[self.viz.layout, layout_widget])
        layout_info = column(children=[self.layout_pie, self.layout_bar])
        self.layout = column(children=[layout_graph, layout_info])

    def bipercent(self, data, col, label=None, mode=None, percents=(0, 100)):
        r"""Bipartite data of given label on given column using given percentages

        Args
        ----
        data : pandas.DataFrame
            Data frame.
        col : str
            Column name to bipartite.
        label : str
            Node label to select from.
        mode : str
            Edge mode to select from.
        percents : (float, float)
            Minimum and maximum partition percentage.

        Returns
        -------
        remain_set : set
            A set of indices to remain.
        remove_set : set
            A set of indices to remove.

        """
        # select data
        assert not (label is not None and mode is not None)
        if label is not None:
            data = data[data['label'] == label][col]
        elif mode is not None:
            data = data[data['mode'] == mode][col]
        else:
            raise RuntimeError()

        # get partition index
        pmin = int(np.floor(len(data) * percents[0] / 100)) - 1
        pmax = int(np.ceil(len(data) * percents[1] / 100)) - 1
        pmin = max(pmin, 0)
        pmax = min(pmax, len(data) - 1)

        # get sorted values
        vals = sorted(data.tolist())

        # get two sets
        remain_set = set(data.index[(vals[pmin] <= data) & (data <= vals[pmax])].tolist())
        remove_set = set(data.index[(data  < vals[pmin]) | (vals[pmax] <  data)].tolist())
        return remain_set, remove_set

    def cut_bins(self, n, data, col, label=None, mode=None):
        r"""Cut given column of given data into given number of bins

        Args
        ----
        n : int
            Number of bins.
        data : pandas.DataFrame
            Data.
        col : str
            Column name to divide.
        label : str
            Node label to select from.
        mode : str
            Edge mode to select from.

        Returns
        -------
        breakpoints : [float, ...]
            (n + 1) breakpoints of bins.

        """
        # select data
        assert not (label is not None and mode is not None)
        if label is not None:
            vals = data[data['label'] == label][col]
        elif mode is not None:
            vals = data[data['mode'] == mode][col]
        else:
            raise RuntimeError()

        # get all break points
        bias = max(-vals.min() * 2, 1) if vals.min() <= 0 else 0
        vals = np.log(vals + bias)
        max_val = vals.max()
        min_val = vals.min()
        scale = (max_val - min_val) / (n - 1)
        breakpoints = [None for i in range(n + 1)]
        breakpoints[0] = min_val
        for i in range(n):
            breakpoints[i + 1] = (0.5 + i) * scale + min_val
        breakpoints[n] = max_val
        breakpoints = np.linspace(min_val, max_val, num=n + 1, endpoint=True).tolist()
        breakpoints = np.exp(breakpoints) - bias
        breakpoints[0]  -= (max_val - min_val)
        breakpoints[-1] += (max_val - min_val)
        return breakpoints.tolist()

    def select(self):
        r"""Select data source based on graph states"""
        # ignore recursive selection
        if self.select_lock != 0:
            return
        else:
            pass

        # ignore empty selection
        if len(self.node_source.selected.indices) == 0:
            return
        else:
            pass

        # lock selection
        self.select_lock += 1

        # get all adjacent neighbors of all selections
        roots = set(self.node_source.selected.indices)
        children = set([])
        for root in roots:
            leaves = self.adj_source[root]
            for _, leaf in leaves:
                children.add(leaf)
        
        # update selections
        self.node_source.selected.indices = list(roots) + list((roots | children) - roots)

        # update the lastest selection statistics
        self.select_to_pie_bar(self.node_source.selected.indices[0])

        # unlock selection
        self.select_lock -= 1

    def select_to_pie_bar(self, idx):
        r"""Update pie and bar chart source by given index
        
        Args
        ----
        idx : int
            Index of selected data source.

        """
        # fresh count buffer
        cnt_n_cont = [0 for i in range(8)]
        cnt_n_coop = [0 for i in range(8)]
        cnt_n_appear = [0 for i in range(8)]
        cnt_n_know = [0 for i in range(8)]

        # update count
        for itr_e, itr_n in self.adj_source[idx]:
            mode = self.edge_data['mode'].iloc[itr_e]
            label = self.node_data['label'].iloc[itr_n]
            edge_iters = dict(
                appear=('cont', self.bin_n_cont, cnt_n_cont),
                know  =('coop', self.bin_n_coop, cnt_n_coop))
            node_iters = dict(
                comic=('#appear', self.bin_n_appear, cnt_n_appear),
                hero =('#know'  , self.bin_n_know  , cnt_n_know  ))
            iter_list = [
                [self.edge_data, itr_e] + list(edge_iters[mode] ),
                [self.node_data, itr_n] + list(node_iters[label])]
            for data, itr, col, bin_n, cnt_n in iter_list:
                val = data[col].iloc[itr]
                for i in range(1, len(bin_n)):
                    if bin_n[i - 1] >= val and (val > bin_n[i]):
                        cnt_n[i - 1] += 1
                        break
                    else:
                        pass

        # project bin count to angle space
        angle_space_n_cont   = [0]
        angle_space_n_coop   = [0]
        angle_space_n_appear = [0]
        angle_space_n_know   = [0]
        iter_list = [
            (cnt_n_cont  , angle_space_n_cont  ),
            (cnt_n_coop  , angle_space_n_coop  ),
            (cnt_n_appear, angle_space_n_appear),
            (cnt_n_know  , angle_space_n_know  )]
        for cnt_n, angle_space_n in iter_list: 
            for val in cnt_n:
                angle_space_n.append(angle_space_n[-1] + val)
            angle_space_n[-1] = max(angle_space_n[-1], 1)
            for i in range(len(angle_space_n)):
                angle_space_n[i] /= angle_space_n[-1]
                angle_space_n[i] *= (2 * np.pi)

        # update the whole data source
        self.pie_source_n_cont.patch(dict(
            start_angle=[(slice(8), angle_space_n_cont[0:8])],
            end_angle=[(slice(8), angle_space_n_cont[1:9])]))
        self.pie_source_n_coop.patch(dict(
            start_angle=[(slice(8), angle_space_n_coop[0:8])],
            end_angle=[(slice(8), angle_space_n_coop[1:9])]))
        self.pie_source_n_appear.patch(dict(
            start_angle=[(slice(8), angle_space_n_appear[0:8])],
            end_angle=[(slice(8), angle_space_n_appear[1:9])]))
        self.pie_source_n_know.patch(dict(
            start_angle=[(slice(8), angle_space_n_know[0:8])],
            end_angle=[(slice(8), angle_space_n_know[1:9])]))
        self.bar_source_n_cont.patch(dict(count=[(slice(8), cnt_n_cont)]))
        self.bar_source_n_coop.patch(dict(count=[(slice(8), cnt_n_coop)]))
        self.bar_source_n_appear.patch(dict(count=[(slice(8), cnt_n_appear)]))
        self.bar_source_n_know.patch(dict(count=[(slice(8), cnt_n_know)]))

    def update(self):
        r"""Update data source based on widget states"""
        # create data mirror for current update
        node_data = self.node_data.copy()
        edge_data = self.edge_data.copy()

        # --------------------------------------------------------------------------------

        # update ratio of node shape
        for label in self.slider_node:
            ratio = np.exp2(self.slider_node[label].value)
            node_data.loc[node_data['label'] == label, ':size'] *= ratio

        # update ratio of edge width
        for mode in self.slider_edge:
            ratio = np.exp2(self.slider_edge[mode].value)
            edge_data.loc[edge_data['mode'] == mode, ':width'] *= ratio

        # also manually update ratio of edge width in personal info legend
        ratio = np.exp2(self.slider_edge['appear'].value)
        new_width = np.linspace(16, 2, num=8, endpoint=True) * ratio
        self.pie_source_n_cont.patch(dict(line_width=[(slice(8), new_width)]))
        ratio = np.exp2(self.slider_edge['know'].value)
        new_width = np.linspace(16, 2, num=8, endpoint=True) * ratio
        self.pie_source_n_coop.patch(dict(line_width=[(slice(8), new_width)]))

        # --------------------------------------------------------------------------------

        # get directly removed nodes by checkbox and range slider
        remain_node_set = set([])
        remove_node_set = set([])
        for i, _label in enumerate(self.checkbox_node.labels):
            label = _label[0].lower() + _label[1:]
            if i in self.checkbox_node.active:
                remain_node_set |= set(node_data.index[node_data['label'] == label].tolist())
            else:
                remove_node_set |= set(node_data.index[node_data['label'] == label].tolist())
        inter1, union1 = self.bipercent(
            node_data, col='sign', label='comic', percents=self.range_slider_sign.value)
        inter2, union2 = self.bipercent(
            node_data, col='pr', label='hero', percents=self.range_slider_pr.value)
        remain_node_set &= (inter1 | inter2)
        remove_node_set |= (union1 | union2)

        # get directly removed edges by checkbox and range slider
        remain_edge_set = set([])
        remove_edge_set = set([])
        for i, _label in enumerate(self.checkbox_edge.labels):
            label = _label[0].lower() + _label[1:]
            if i in self.checkbox_edge.active:
                remain_edge_set |= set(edge_data.index[edge_data['mode'] == label].tolist())
            else:
                remove_edge_set |= set(edge_data.index[edge_data['mode'] == label].tolist())
        inter1, union1 = self.bipercent(
            edge_data, col='cont', mode='appear', percents=self.range_slider_cont.value)
        inter2, union2 = self.bipercent(
            edge_data, col='coop', mode='know', percents=self.range_slider_coop.value)
        remain_edge_set &= (inter1 | inter2)
        remove_edge_set |= (union1 | union2)

        # --------------------------------------------------------------------------------

        # get nodes that all its neighbors still existing
        node_set1 = set(node_data['label_name'][remain_node_set].tolist())
        node_set2 = set(edge_data['label_name1'][remain_edge_set].tolist())
        node_set3 = set(edge_data['label_name2'][remain_edge_set].tolist())
        node_set = node_set1 & (node_set2 | node_set3)

        # filter unsatisfying nodes and edges
        filter1 = node_data['label_name'].isin(node_set)
        filter2 = edge_data['label_name1'].isin(node_set)
        filter3 = edge_data['label_name2'].isin(node_set)
        remain_node_set &= set(node_data.index[filter1].tolist())
        remove_node_set |= set(node_data.index[~filter1].tolist())
        remain_edge_set &= set(edge_data.index[filter2 & filter3].tolist())
        remove_edge_set |= set(edge_data.index[~(filter2 & filter3)].tolist())

        # get non-zero degree nodes by remaining edges
        node_set1 = set(edge_data['label_name1'][remain_edge_set].tolist())
        node_set2 = set(edge_data['label_name2'][remain_edge_set].tolist())
        node_set = (node_set1 | node_set2)

        # remove zero degree nodes
        filter1 = node_data['label_name'].isin(node_set)
        remain_node_set &= set(node_data.index[filter1].tolist())
        remove_node_set |= set(node_data.index[~filter1].tolist())

        # lighten removed nodes and edges
        light_alpha = 0.05
        node_data.loc[remove_node_set, ':fill_alpha'] = light_alpha
        node_data.loc[remove_node_set, ':line_alpha'] = light_alpha
        node_data.loc[remove_node_set, ':sel:fill_alpha'] = light_alpha
        node_data.loc[remove_node_set, ':sel:line_alpha'] = light_alpha
        node_data.loc[remove_node_set, ':nosel:fill_alpha'] = light_alpha
        node_data.loc[remove_node_set, ':nosel:line_alpha'] = light_alpha
        node_data.loc[remove_node_set, ':hover:fill_alpha'] = light_alpha
        edge_data.loc[remove_edge_set, ':line_alpha'] = light_alpha
        edge_data.loc[remove_edge_set, ':sel:line_alpha'] = light_alpha
        edge_data.loc[remove_edge_set, ':nosel:line_alpha'] = light_alpha
        edge_data.loc[remove_edge_set, ':hover:line_alpha'] = light_alpha

        # --------------------------------------------------------------------------------

        # update the whole data source
        self.node_source.data = node_data.to_dict(orient='list')
        self.edge_source.data = edge_data.to_dict(orient='list')

    def click(self, *args, **kargs):
        r"""Update selection by bar chart label clicking"""
        print(args)
        print(kargs)

    def checkbox_(self):
        r"""Deploy checkbox"""
        # generate widgets
        self.checkbox_title_node = Div(width=self.CBOX_W, height=self.CBOX_H_TITLE, text='Node')
        self.checkbox_title_edge = Div(width=self.CBOX_W, height=self.CBOX_H_TITLE, text='Edge')

        self.checkbox_node = CheckboxGroup(
                width=self.CBOX_W, height=self.CBOX_H, labels=['Comic', 'Hero'], active=[0, 1])
        self.checkbox_edge = CheckboxGroup(
                width=self.CBOX_W, height=self.CBOX_H, labels=['Appear', 'Know'], active=[0, 1])

        # set change hook
        self.checkbox_node.on_change('active', lambda attr, old, new: self.update())
        self.checkbox_edge.on_change('active', lambda attr, old, new: self.update())

        # configure layout
        self.layout_checkbox = row(
            children=[
                column(children=[self.checkbox_title_node, self.checkbox_node]),
                column(children=[self.checkbox_title_edge, self.checkbox_edge])])

    def slider_(self):
        r"""Deploy slider"""
        # generate widgets
        self.slider_node, self.slider_edge = dict(), dict()
        self.slider_node['comic'] = Slider(
            width=self.SLDR_W, height=self.SLDR_H, title='Node Size (Comic)',
            start=-2, end=2, step=0.1, value=0)
        self.slider_node['hero'] = Slider(
            width=self.SLDR_W, height=self.SLDR_H, title='Node Size (Hero)',
            start=-2, end=2, step=0.1, value=0)
        self.slider_edge['appear'] = Slider(
            width=self.SLDR_W, height=self.SLDR_H, title='Edge Width (Appear)',
            start=-2, end=2, step=0.1, value=0)
        self.slider_edge['know'] = Slider(
            width=self.SLDR_W, height=self.SLDR_H, title='Edge Width (Know)',
            start=-2, end=2, step=0.1, value=0)

        # set change hook
        self.slider_node['comic' ].on_change('value', lambda attr, old, new: self.update())
        self.slider_node['hero'  ].on_change('value', lambda attr, old, new: self.update())
        self.slider_edge['appear'].on_change('value', lambda attr, old, new: self.update())
        self.slider_edge['know'  ].on_change('value', lambda attr, old, new: self.update())

        # configure layout
        self.layout_slider = column(
            children=[
                self.slider_node['comic'], self.slider_node['hero'], self.slider_edge['appear'],
                self.slider_edge['know']])

    def range_slider_(self):
        r"""Deploy range slider"""
        # generate widgets
        self.range_slider_sign = RangeSlider(
            width=self.RNG_SLDR_W, height=self.RNG_SLDR_H, title='Node Percentage (Comic)',
            start=0, end=100, step=99/len(self.node_data), value=(0, 100))
        self.range_slider_pr = RangeSlider(
            width=self.RNG_SLDR_W, height=self.RNG_SLDR_H, title='Node Percentage (Hero)',
            start=0, end=100, step=99/len(self.node_data), value=(0, 100))
        self.range_slider_cont = RangeSlider(
            width=self.RNG_SLDR_W, height=self.RNG_SLDR_H, title='Edge Percentage (Appear)',
            start=0, end=100, step=99/len(self.edge_data), value=(0, 100))
        self.range_slider_coop = RangeSlider(
            width=self.RNG_SLDR_W, height=self.RNG_SLDR_H, title='Edge Percentage (Know)',
            start=0, end=100, step=99/len(self.edge_data), value=(0, 100))

        # set change hook
        self.range_slider_sign.on_change('value', lambda attr, old, new: self.update())
        self.range_slider_pr  .on_change('value', lambda attr, old, new: self.update())
        self.range_slider_cont.on_change('value', lambda attr, old, new: self.update())
        self.range_slider_coop.on_change('value', lambda attr, old, new: self.update())

        # configure layout
        self.layout_range_slider = column(
            children=[
                self.range_slider_sign, self.range_slider_pr, self.range_slider_cont,
                self.range_slider_coop])

    def bin_(self):
        """Deploy bins of personal info charts"""
        # divide pie chart space into bins
        self.bin_n_cont = self.cut_bins(8, data=self.edge_data, col='cont', mode='appear')
        self.bin_n_coop = self.cut_bins(8, data=self.edge_data, col='coop', mode='know')
        self.bin_n_appear = self.cut_bins(8, data=self.node_data, col='#appear', label='comic')
        self.bin_n_know = self.cut_bins(8, data=self.node_data, col='#know', label='hero')
        self.bin_n_sign = self.cut_bins(8, data=self.node_data, col='sign', label='comic')
        self.bin_n_pr = self.cut_bins(8, data=self.node_data, col='pr', label='hero')
        self.bin_n_cont.reverse()
        self.bin_n_coop.reverse()
        self.bin_n_appear.reverse()
        self.bin_n_know.reverse()
        self.bin_n_sign.reverse()
        self.bin_n_pr.reverse()

    def pie_(self):
        r"""Deploy pie chart"""
        # allocate pie source
        equal_space = np.linspace(0, 2 * np.pi, num=9, endpoint=True)
        self.pie_source_n_cont = ColumnDataSource(dict(
            xs=[(0, 0)] * 8, ys=[(0, 0)] * 8, line_width=np.linspace(16, 2, num=8, endpoint=True),
            start_angle=equal_space[0:8], end_angle=equal_space[1:9],
            fill_color=['black'] * 8, fill_alpha=np.linspace(0.85, 0.35, num=8, endpoint=True),
            legend=["<= {:.2E}".format(Decimal(self.bin_n_cont[i])) for i in range(8)]))
        self.pie_source_n_coop = ColumnDataSource(dict(
            xs=[(0, 0)] * 8, ys=[(0, 0)] * 8, line_width=np.linspace(16, 2, num=8, endpoint=True),
            start_angle=equal_space[0:8], end_angle=equal_space[1:9],
            fill_color=['black'] * 8, fill_alpha=np.linspace(0.85, 0.35, num=8, endpoint=True),
            legend=["<= {:.2f}".format(self.bin_n_coop[i]) for i in range(8)]))
        self.pie_source_n_appear = ColumnDataSource(dict(
            start_angle=equal_space[0:8], end_angle=equal_space[1:9], fill_color=BuGn9[0:8],
            legend=["<= {:.2f}".format(self.bin_n_appear[i]) for i in range(8)]))
        self.pie_source_n_know = ColumnDataSource(dict(
            start_angle=equal_space[0:8], end_angle=equal_space[1:9], fill_color=YlOrRd9[0:8],
            legend=["<= {:.2f}".format(self.bin_n_know[i]) for i in range(8)]))

        # create canvas
        fig_n_cont = figure(
            width=self.PIE_W, height=self.PIE_H,
            title='Contribution Dist. in \"Comics Appears\"',
            x_range=(-1.2, 3.2), y_range=(-1.2, 1.2), toolbar_location=None)
        fig_n_coop = figure(
            width=self.PIE_W, height=self.PIE_H,
            title='Cooperation Dist. with \"Heros Knows\"',
            x_range=(-1.2, 3.2), y_range=(-1.2, 1.2), toolbar_location=None)
        fig_n_appear = figure(
            width=self.PIE_W, height=self.PIE_H,
            title='#Appear Dist. of \"Comics Appears\"',
            x_range=(-1.2, 3.2), y_range=(-1.2, 1.2), toolbar_location=None)
        fig_n_know = figure(
            width=self.PIE_W, height=self.PIE_H,
            title='#Know Dist. of \"Heros Knows\"',
            x_range=(-1.2, 3.2), y_range=(-1.2, 1.2), toolbar_location=None)
        fig_n_cont.axis.visible = False
        fig_n_coop.axis.visible = False
        fig_n_appear.axis.visible = False
        fig_n_know.axis.visible = False
        fig_n_cont.grid.visible = False
        fig_n_coop.grid.visible = False
        fig_n_appear.grid.visible = False
        fig_n_know.grid.visible = False

        # specify edge related legend
        fig_n_cont.multi_line(
            xs='xs', ys='ys',  line_width='line_width',
            line_color='fill_color', line_alpha='fill_alpha',
            source=self.pie_source_n_cont, legend='legend')
        fig_n_coop.multi_line(
            xs='xs', ys='ys',  line_width='line_width',
            line_color='fill_color', line_alpha='fill_alpha',
            source=self.pie_source_n_coop, legend='legend')

        # create pie chart
        pie_n_cont = fig_n_cont.wedge(
            x=0, y=0, radius=1, start_angle='start_angle', end_angle='end_angle',
            fill_color='fill_color', fill_alpha='fill_alpha', line_color='white',
            source=self.pie_source_n_cont)
        pie_n_coop = fig_n_coop.wedge(
            x=0, y=0, radius=1, start_angle='start_angle', end_angle='end_angle',
            fill_color='fill_color', fill_alpha='fill_alpha', line_color='white',
            source=self.pie_source_n_coop)
        pie_n_appear = fig_n_appear.wedge(
            x=0, y=0, radius=1, start_angle='start_angle', end_angle='end_angle',
            fill_color='fill_color', source=self.pie_source_n_appear, legend='legend',
            line_color='white')
        pie_n_know = fig_n_know.wedge(
            x=0, y=0, radius=1, start_angle='start_angle', end_angle='end_angle',
            fill_color='fill_color', source=self.pie_source_n_know, legend='legend',
            line_color='white')

        # reset legend location
        fig_n_cont.legend.location = 'center_right'
        fig_n_coop.legend.location = 'center_right'
        fig_n_appear.legend.location = 'center_right'
        fig_n_know.legend.location = 'center_right'

        # configure layout
        self.layout_pie = row(children=[fig_n_appear, fig_n_cont, fig_n_know, fig_n_coop])

    def bar_(self):
        r"""Deploy bar chart"""
        # allocate pie source
        zero_space = np.zeros(shape=(8,))
        self.bar_source_n_cont = ColumnDataSource(dict(
            count=zero_space, fill_color=['black'] * 8,
            fill_alpha=np.linspace(0.85, 0.35, num=8, endpoint=True),
            name=["<= {:2E}".format(Decimal(self.bin_n_cont[i])) for i in range(8)]))
        self.bar_source_n_coop = ColumnDataSource(dict(
            count=zero_space, fill_color=['black'] * 8,
            fill_alpha=np.linspace(0.85, 0.35, num=8, endpoint=True),
            name=["<= {:.2f}".format(self.bin_n_coop[i]) for i in range(8)]))
        self.bar_source_n_appear = ColumnDataSource(dict(
            count=zero_space, fill_color=BuGn9[0:8],
            name=["<= {:.2f}".format(self.bin_n_appear[i]) for i in range(8)]))
        self.bar_source_n_know = ColumnDataSource(dict(
            count=zero_space, fill_color=YlOrRd9[0:8],
            name=["<= {:.2f}".format(self.bin_n_know[i]) for i in range(8)]))

        # create canvas
        fig_n_cont = figure(
            width=self.BAR_W, height=self.BAR_H,
            tools="ypan,tap,ywheel_zoom", active_scroll='ywheel_zoom',
            title='Contribution Count in \"Comics Appears\"',
            x_range=self.bar_source_n_cont.data['name'], y_range=Range1d(0, 100, bounds='auto'),
            toolbar_location=None)
        fig_n_coop = figure(
            width=self.BAR_W, height=self.BAR_H,
            tools="ypan,tap,ywheel_zoom", active_scroll='ywheel_zoom',
            title='Cooperation Count with \"Heros Knows\"',
            x_range=self.bar_source_n_coop.data['name'], y_range=Range1d(0, 100, bounds='auto'),
            toolbar_location=None)
        fig_n_appear = figure(
            width=self.BAR_W, height=self.BAR_H,
            tools="ypan,tap,ywheel_zoom", active_scroll='ywheel_zoom',
            title='#Appear Count of \"Comics Appears\"',
            x_range=self.bar_source_n_appear.data['name'], y_range=Range1d(0, 100, bounds='auto'),
            toolbar_location=None)
        fig_n_know = figure(
            width=self.BAR_W, height=self.BAR_H,
            tools="ypan,tap,ywheel_zoom", active_scroll='ywheel_zoom',
            title='#Know Count of \"Heros Knows\"',
            x_range=self.bar_source_n_know.data['name'], y_range=Range1d(0, 100, bounds='auto'),
            toolbar_location=None)
        fig_n_cont.xaxis.visible = False
        fig_n_coop.xaxis.visible = False
        fig_n_appear.xaxis.visible = False
        fig_n_know.xaxis.visible = False
        fig_n_cont.xgrid.visible = False
        fig_n_coop.xgrid.visible = False
        fig_n_appear.xgrid.visible = False
        fig_n_know.xgrid.visible = False

        # create bar chart
        bar_n_cont = fig_n_cont.vbar(
            x='name', top='count', width=0.85, fill_color='fill_color', fill_alpha='fill_alpha',
            line_color='white', source=self.bar_source_n_cont)
        bar_n_coop = fig_n_coop.vbar(
            x='name', top='count', width=0.85, fill_color='fill_color', fill_alpha='fill_alpha',
            line_color='white', source=self.bar_source_n_coop)
        bar_n_appear = fig_n_appear.vbar(
            x='name', top='count', width=0.85, fill_color='fill_color', line_color='white',
            source=self.bar_source_n_appear)
        bar_n_know = fig_n_know.vbar(
            x='name', top='count', width=0.85, fill_color='fill_color', line_color='white',
            source=self.bar_source_n_know)
        
        # create label
        label_n_cont = LabelSet(
            x='name', y='count', text='count', level='glyph', x_offset=0, y_offset=0,
            source=self.bar_source_n_cont, render_mode='canvas', text_align='center')
        label_n_coop = LabelSet(
            x='name', y='count', text='count', level='glyph', x_offset=0, y_offset=0,
            source=self.bar_source_n_coop, render_mode='canvas', text_align='center')
        label_n_appear = LabelSet(
            x='name', y='count', text='count', level='glyph', x_offset=0, y_offset=0,
            source=self.bar_source_n_appear, render_mode='canvas', text_align='center')
        label_n_know = LabelSet(
            x='name', y='count', text='count', level='glyph', x_offset=0, y_offset=0,
            source=self.bar_source_n_know, render_mode='canvas', text_align='center')

        # set change hook
        self.bar_source_n_cont.selected.on_change(
            'indices', lambda attr, old, new: self.click())
        self.bar_source_n_coop.selected.on_change(
            'indices', lambda attr, old, new: self.click())
        self.bar_source_n_appear.selected.on_change(
            'indices', lambda attr, old, new: self.click())
        self.bar_source_n_know.selected.on_change(
            'indices', lambda attr, old, new: self.click())

        # configure layout
        fig_n_cont.add_layout(label_n_cont)
        fig_n_coop.add_layout(label_n_coop)
        fig_n_appear.add_layout(label_n_appear)
        fig_n_know.add_layout(label_n_know)
        self.layout_bar = row(children=[fig_n_appear, fig_n_cont, fig_n_know, fig_n_coop])

class BiGraphWidget(object):
    # constants
    BTN_W = 140
    BTN_H = 20

    TXT_W = 284
    TXT_H = 30

    DIV_W = 300
    DIV_LINE_H = 10

    IMG_W = 300
    IMG_H = 300

    DIST_W = 300
    DIST_H = 300

    def __init__(self, major_inter, minor_inter):
        r"""Initialize the class

        Args
        ----
        major_inter : GraphWidget
            Interactive graph and widget visualization of major network.
        minor_inter : GraphWidget
            Interactive graph and widget visualization of minor network.

        """
        # aggregate two graph layout
        self.major_inter = major_inter
        self.minor_inter = minor_inter
        self.inter_dict = dict(major=self.major_inter, minor=self.minor_inter)

        # deploy selection's trigger
        self.select_lock = False
        self.major_inter.node_source.selected.on_change(
            'indices', lambda attr, old, new: self.select('major'))
        self.minor_inter.node_source.selected.on_change(
            'indices', lambda attr, old, new: self.select('minor'))

        # deploy tab
        self.layout_tab = Tabs(tabs=[
            Panel(child=self.major_inter.layout, title="Majority"),
            Panel(child=self.minor_inter.layout, title="Minority")])

        # deploy text input
        self.find_buffer, self.find_ptr = [], 0
        self.input_()

        # deploy search tool and image
        self.search_engine = google_images_download.googleimagesdownload()
        self.img_()

        # deploy info div
        self.div_()

        # deploy dist plot
        self.dist_()

        # configure layout
        layout_info = column(
            children=[self.layout_input, self.layout_img, self.layout_div, self.layout_dist])
        self.layout = row(layout_info, self.layout_tab)

    def find(self, *args, **kargs):
        r"""Find the most similar node from all nodes"""
        # get smilarity
        self.find_buffer = []
        for part in ('major', 'minor'):
            node_data = self.inter_dict[part].node_data
            for i in range(len(node_data)):
                name = node_data.loc[i, '_name']
                ratio = fuzz.partial_ratio(self.input.value, name)
                self.find_buffer.append((part, i, name, ratio))
        self.find_buffer = sorted(self.find_buffer, reverse=True, key=lambda x: x[-1])
        self.find_ptr = 0

        # activate selection
        tab_id = dict(major=0, minor=1)
        self.layout_tab.active = tab_id[self.find_buffer[self.find_ptr][0]]
        viz_inter = self.inter_dict[self.find_buffer[self.find_ptr][0]]
        root = self.find_buffer[self.find_ptr][1]
        branches = [itr for itr, _ in viz_inter.adj_source[root]]
        leaves = [itr for _, itr in viz_inter.adj_source[root]]
        viz_inter.node_source.selected.indices = [root]
        viz_inter.edge_source.selected.indices = branches

    def find_next(self, *args, **kargs):
        r"""Goto next search item in buffer"""
        # check validation
        if len(self.find_buffer) == 0 or \
           self.find_buffer[self.find_ptr][-1] < 80:
            return
        else:
            self.find_ptr += 1

        if (self.find_ptr >= len(self.find_buffer)) or \
           self.find_buffer[self.find_ptr][-1] < 80:
            return
        else:
            pass

        # activate selection
        tab_id = dict(major=0, minor=1)
        self.layout_tab.active = tab_id[self.find_buffer[self.find_ptr][0]]
        viz_inter = self.inter_dict[self.find_buffer[self.find_ptr][0]]
        root = self.find_buffer[self.find_ptr][1]
        branches = [itr for itr, _ in viz_inter.adj_source[root]]
        leaves = [itr for _, itr in viz_inter.adj_source[root]]
        viz_inter.node_source.selected.indices = [root]
        viz_inter.edge_source.selected.indices = branches

    def select(self, viz_part):
        r"""Select data source based on graph states
        
        Args
        ----
        viz_part : str
            Tab string identifier being selected.

        """
        # focus on active tab
        viz_inter = self.inter_dict[viz_part]
        node_data = viz_inter.node_data
        edge_data = viz_inter.edge_data

        # ignore recursive selection
        if viz_inter.select_lock != 1:
            return
        else:
            pass

        # lock selection
        self.select_lock += 1

        # chech selection status
        if len(viz_inter.node_source.selected.indices) == 0:
            self.div_dict['name'].text = '(non-selection)'
            return
        else:
            pass

        # appending action
        ind = viz_inter.node_source.selected.indices[0]
        name = node_data.loc[ind, '_name']
        sign = node_data.loc[ind, 'sign']
        pr = node_data.loc[ind, 'pr']
        n_appear = node_data.loc[ind, '#appear']
        n_know = node_data.loc[ind, '#know']
        search = re.split(r'\s*[^0-9a-zA-Z\s-]+\s*', name)[0]
        self.image_url_data.data = dict(url=[self.search_engine.download(
            dict(keywords="Marvel Comic {}".format(search), limit=1))])
        self.div_dict['name'].text = name
        self.div_dict['sign'].text = "{:.8f}".format(sign)
        self.div_dict['pr'].text = "{:.8f}".format(pr)
        self.div_dict['n_appear'].text = "{:d}".format(int(n_appear))
        self.div_dict['n_know'].text = "{:d}".format(int(n_know))

        # unlock selection
        self.select_lock -= 1

    def input_(self):
        r"""Deploy name input for searching"""
        # generate widgets
        self.button_search = Button(
            width=self.BTN_W, height=self.BTN_H, label='Search', button_type='success')
        self.button_next = Button(
            width=self.BTN_W, height=self.BTN_H, label='Next', button_type='success')
        self.input = TextInput(
            width=self.TXT_W, height=self.TXT_H, value='', title='Searching Hero/Comic Name')

        # set change hook
        self.button_search.on_click(self.find)
        self.button_next.on_click(self.find_next)

        # configure layout
        self.layout_input = column(self.input, row(self.button_search, self.button_next))

    def img_(self):
        r"""Deploy image"""
        # allocate url source
        self.image_url_data = ColumnDataSource(dict(url=['(non-selection)']))

        # create canvas
        fig_image = figure(
            width=self.IMG_W, height=self.IMG_H,
            x_range=(-self.IMG_W // 2 - 1, self.IMG_W // 2 + 1),
            y_range=(-self.IMG_H // 2 - 1, self.IMG_H // 2 + 1),
            toolbar_location=None)
        fig_image.axis.visible = False
        fig_image.grid.visible = False

        # create image
        image_content = fig_image.image_url(
            x=0, y=0, w=None, h=None, anchor="center", url='url', source=self.image_url_data)

        # configure layout
        self.layout_img = fig_image

    def div_(self):
        r"""Deploy info div"""
        # generate widgets
        name_title = Div(
            width=self.DIV_W, height=self.DIV_LINE_H,
            text='<b>Hero/Comic Name</b>')
        name_content = Div(
            width=self.DIV_W, height=self.DIV_LINE_H,
            text='(non-selection)')
        sign_title = Div(
            width=self.DIV_W // 2, height=self.DIV_LINE_H,
            text='<b>PageRank</b>')
        sign_content = Div(
            width=self.DIV_W // 2, height=self.DIV_LINE_H,
            text='(non-selection)')
        pr_title = Div(
            width=self.DIV_W // 2, height=self.DIV_LINE_H,
            text='<b>PageRank</b>')
        pr_content = Div(
            width=self.DIV_W // 2, height=self.DIV_LINE_H,
            text='(non-selection)')
        n_appear_title = Div(
            width=self.DIV_W // 2, height=self.DIV_LINE_H,
            text='<b>#Appear</b>')
        n_appear_content = Div(
            width=self.DIV_W // 2, height=self.DIV_LINE_H,
            text='(non-selection)')
        n_know_title = Div(
            width=self.DIV_W // 2, height=self.DIV_LINE_H,
            text='<b>#Know</b>')
        n_know_content = Div(
            width=self.DIV_W // 2, height=self.DIV_LINE_H,
            text='(non-selection)')

        # broadcast changable widgets
        self.div_dict = dict(
            name=name_content, sign=sign_content, pr=pr_content, n_appear=n_appear_content,
            n_know=n_know_content)

        # configure layout
        name_info = column(name_title, name_content)
        sign_info = column(sign_title, sign_content)
        pr_info = column(pr_title, pr_content)
        n_appear_info = column(n_appear_title, n_appear_content)
        n_know_info = column(n_know_title, n_know_content)
        self.layout_div = column(
            name_info, row(sign_info, pr_info), row(n_appear_info, n_know_info))

    def cut_count_node_bins(self, n, col, label):
        """Cut and count given column of all node data into given number of bins

        Args
        ----
        n : int
            Number of bins.
        col : str
            Column name to divide.
        label : str
            Node label to select from.

        Returns
        -------
        edges : [float, ...]
            A list of lower bound of each bins.
        count : [int, ...]
            Count of samples in each bin.

        """
        # get all data
        data1 = self.major_inter.node_data
        data2 = self.minor_inter.node_data
        data1 = data1[data1['label'] == label]
        data2 = data2[data2['label'] == label]
        data1 = data1[col].values
        data2 = data2[col].values
        data = np.concatenate([data1, data2], axis=0)

        # get bins
        log_data = np.log(data)
        max_val = log_data.max()
        min_val = log_data.min()
        scale = (max_val - min_val) / (n - 1)
        breakpoints = [None for i in range(n + 1)]
        breakpoints[0] = min_val
        for i in range(n):
            breakpoints[i + 1] = (0.5 + i) * scale + min_val
        breakpoints[n] = max_val
        breakpoints = np.linspace(min_val, max_val, num=n + 1, endpoint=True).tolist()
        breakpoints[0]  -= (max_val - min_val)
        breakpoints[-1] += (max_val - min_val)
        breakpoints.reverse()
        breakpoints = np.exp(breakpoints).tolist()

        # count bins
        count = [0 for i in range(len(breakpoints) - 1)]
        for i in range(len(count)):
            count[i] = ((breakpoints[i] >= data) & (data > breakpoints[i + 1])).sum()
        return breakpoints, count

    def dist_(self):
        r"""Deploy dist plot"""
        # get bins and counts
        sign_edge, sign_count = self.cut_count_node_bins(10, col='sign', label='comic')
        pr_edge  , pr_count   = self.cut_count_node_bins(10, col='pr'  , label='hero')
        sign_avg = [np.sqrt(sign_edge[i] * sign_edge[i + 1]) for i in range(len(sign_count))]
        pr_avg   = [np.sqrt(pr_edge[i]   * pr_edge[i + 1]  ) for i in range(len(pr_count))]

        # transfer to data source
        source_sign = ColumnDataSource(dict(
            count=sign_count, fill_color=Plasma11[0:10],
            name=["{:.2E}".format(Decimal(itr)) for itr in sign_avg]))
        source_pr = ColumnDataSource(dict(
            count=pr_count, fill_color=Viridis11[0:10],
            name=["{:.2E}".format(Decimal(itr)) for itr in pr_avg]))

        # create canvas
        fig_sign = figure(
            width=self.DIST_W, height=self.DIST_H, title='Comic Significance Distribution Plot',
            y_range=source_sign.data['name'],
            x_range=(0, int(max(source_sign.data['count']) * 1.1)),
            toolbar_location=None)
        fig_pr = figure(
            width=self.DIST_W, height=self.DIST_H, title='Hero PageRank Distribution Plot',
            y_range=source_pr.data['name'],
            x_range=(0, int(max(source_pr.data['count']) * 1.1)),
            toolbar_location=None)

        # bar and line plot
        bar_sign = fig_sign.hbar(
            y='name', right='count', height=0.85, fill_color='fill_color', line_color='white',
            source=source_sign)
        bar_pr = fig_pr.hbar(
            y='name', right='count', height=0.85, fill_color='fill_color', line_color='white',
            source=source_pr)
        line_sign = fig_sign.line(y='name', x='count', line_color='black', source=source_sign)
        line_pr = fig_pr.line(y='name', x='count', line_color='black', source=source_pr)

        # reset legend location
        # fig_sign.legend.location = 'center_right'
        # fig_pr.legend.location = 'center_right'

        # configure layout
        self.layout_dist = column(fig_sign, fig_pr)