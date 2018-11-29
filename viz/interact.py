import numpy as np
from bokeh.models import ColumnDataSource
# Plot, Title, Range1d, LegendItem, Legend
from bokeh.plotting import figure
from bokeh.palettes import *
from bokeh.models import Wedge
from bokeh.models.widgets import Div, CheckboxGroup, Slider, RangeSlider
from bokeh.layouts import row, column
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
    PIE_H = 300

    BAR_W = 300
    BAR_H = 300

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
        self.select_lock = False
        self.node_source.selected.on_change('indices', lambda attr, old, new: self.select())

        # deploy checkbox and its trigger
        self.checkbox_()

        # deploy slider and its trigger
        self.slider_()

        # deploy slider and its trigger
        self.range_slider_()

        # deploy pie and bar chart
        self.pie_()
        self.bar_()

        # configure layout
        layout_widget = column(
            children=[self.layout_checkbox, self.layout_slider, self.layout_range_slider])
        layout_graph = row(children=[self.viz.layout, layout_widget])
        # layout_info = column(children=[self.layout_pie, self.layout_bar])
        layout_info = self.layout_pie
        self.layout = row(children=[layout_info, layout_graph])

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
        max_val = vals.max()
        min_val = vals.min()
        scale = (max_val - min_val) / (n - 1)
        breakpoints = [None for i in range(n + 1)]
        breakpoints[0] = min_val - (max_val - min_val) * 0.01
        for i in range(n):
            breakpoints[i + 1] = (0.5 + i) * scale + min_val
        breakpoints[n] = max_val
        breakpoints = np.linspace(min_val, max_val, num=n + 1, endpoint=True).tolist()
        breakpoints[-1] += (max_val - min_val) * 0.01
        return breakpoints

    def select(self):
        r"""Select data source based on graph states"""
        # ignore recursive selection
        if self.select_lock:
            return
        else:
            pass

        # ignore empty selection
        if len(self.node_source.selected.indices) == 0:
            return
        else:
            pass

        # lock selection
        self.select_lock = True

        # update the lastest selection statistics
        self.select_to_pie(self.node_source.selected.indices[-1])

        # get all adjacent neighbors of all selections
        roots = set(self.node_source.selected.indices)
        children = set([])
        for root in roots:
            leaves = self.adj_source[root]
            for leaf in leaves:
                children.add(leaf)
        
        # update selections
        self.node_source.selected.indices = list(roots | children)

        # unlock selection
        self.select_lock = False

    def select_to_pie(self, idx):
        r"""Update pie chart source by given index
        
        Args
        ----
        idx : int
            Index of selected data source.

        """
        # fresh count buffer
        cnt_n_appear = [0 for i in range(8)]
        cnt_n_know = [0 for i in range(8)]

        # update count
        for itr in self.adj_source[idx]:
            label = self.node_data['label'].iloc[itr]
            if label == 'comic':
                val = self.node_data['#appear'].iloc[itr]
                for i in range(1, 9):
                    if self.bin_n_appear[i - 1] >= val and val > self.bin_n_appear[i]:
                        cnt_n_appear[i - 1] += 1
                    else:
                        pass
            elif label == 'hero':
                val = self.node_data['#know'].iloc[itr]
                for i in range(1, 9):
                    if self.bin_n_know[i - 1] >= val and val > self.bin_n_know[i]:
                        cnt_n_know[i - 1] += 1
                    else:
                        pass
            else:
                raise RuntimeError()

        # project bin count to angle space
        angle_space_n_appear = [0]
        angle_space_n_know = [0]
        for val in cnt_n_appear:
            angle_space_n_appear.append(angle_space_n_appear[-1] + val)
        angle_space_n_appear[-1] = max(angle_space_n_appear[-1], 1)
        for val in cnt_n_know:
            angle_space_n_know.append(angle_space_n_know[-1] + val)
        angle_space_n_know[-1] = max(angle_space_n_know[-1], 1)
        for i in range(len(angle_space_n_appear)):
            angle_space_n_appear[i] /= angle_space_n_appear[-1]
            angle_space_n_appear[i] *= (2 * np.pi)
        for i in range(len(angle_space_n_know)):
            angle_space_n_know[i] /= angle_space_n_know[-1]
            angle_space_n_know[i] *= (2 * np.pi)

        # update the whole data source
        self.pie_source_n_appear.patch(dict(
            start_angle=[(slice(8), angle_space_n_appear[0:8])],
            end_angle=[(slice(8), angle_space_n_appear[1:9])]))
        self.pie_source_n_know.patch(dict(
            start_angle=[(slice(8), angle_space_n_know[0:8])],
            end_angle=[(slice(8), angle_space_n_know[1:9])]))

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

    def pie_(self):
        r"""Deploy pie chart"""
        # divide pie chart space into bins
        self.bin_n_appear = self.cut_bins(8, data=self.node_data, col='#appear', label='comic')
        self.bin_n_know = self.cut_bins(8, data=self.node_data, col='#know', label='hero')
        self.bin_n_appear.reverse()
        self.bin_n_know.reverse()

        # allocate pie source
        equal_space = np.linspace(0, 2 * np.pi, num=9, endpoint=True)
        self.pie_source_n_appear = ColumnDataSource(dict(
            start_angle=equal_space[0:8], end_angle=equal_space[1:9], fill_color=Greens9[0:8],
            legend=["<= {:.1f}".format(self.bin_n_appear[i]) for i in range(8)]))
        self.pie_source_n_know = ColumnDataSource(dict(
            start_angle=equal_space[0:8], end_angle=equal_space[1:9], fill_color=Reds9[0:8],
            legend=["<= {:.1f}".format(self.bin_n_know[i]) for i in range(8)]))

        # create canvas
        fig_n_appear = figure(
            width=self.PIE_W, height=self.PIE_H, title='Comic #Appear Dist.',
            x_range=(-1.2, 2.2), y_range=(-1.2, 1.2), toolbar_location=None)
        fig_n_know = figure(
            width=self.PIE_W, height=self.PIE_H, title='Hero #Know Dist.',
            x_range=(-1.2, 2.2), y_range=(-1.2, 1.2), toolbar_location=None)
        fig_n_appear.axis.visible = False
        fig_n_know.axis.visible = False

        # create pie chart
        pie_n_appear = fig_n_appear.wedge(
            x=0, y=0, radius=1, start_angle='start_angle', end_angle='end_angle',
            fill_color='fill_color', source=self.pie_source_n_appear, legend='legend',
            line_color='white')
        pie_n_know = fig_n_know.wedge(
            x=0, y=0, radius=1, start_angle='start_angle', end_angle='end_angle',
            fill_color='fill_color', source=self.pie_source_n_know, legend='legend',
            line_color='white')

        # reset legend location
        fig_n_appear.legend.location = 'center_right'
        fig_n_know.legend.location = 'center_right'

        # configure layout
        self.layout_pie = column(children=[fig_n_appear, fig_n_know])

    def bar_(self):
        r"""Deploy bar chart"""
        pass