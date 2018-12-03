import networkx as nx
import numpy as np
import pandas as pd
from bokeh.models import ColumnDataSource, Plot, Range1d, GlyphRenderer
from bokeh.models.graphs import from_networkx
from bokeh.palettes import *
from bokeh.models import MultiLine, Scatter
from bokeh.models.graphs import NodesAndLinkedEdges, EdgesAndLinkedNodes
from bokeh.models import PanTool, WheelZoomTool, TapTool, HoverTool


class GraphViz(object):
    r"""Graph Visualization"""
    # constants
    PLOT_W = 900
    PLOT_H = 600

    def __init__(self, graph):
        r"""Initialize the class

        Args
        ----
        graph : networkx.Graph
            The graph to visualize.

        """
        # initialize layout framework
        self.layout = None

        # get graph plot
        self.from_graph(graph)
        self.layout = self.plot

    def get_graph_pos(self, graph, col='pos', alpha=0.1):
        r"""Get graph position related data

        Args
        ----
        col : str
            Attribute name of node holding its position.
        alpha : float
            Range extension rate.

        """
        # get position data
        pos_dict = nx.get_node_attributes(graph, col)
        pos_array = np.array([itr for itr in pos_dict.values()])

        # get x-axis and y-axis boundaries
        xmin, xmax = pos_array[:, 0].min(), pos_array[:, 0].max()
        ymin, ymax = pos_array[:, 1].min(), pos_array[:, 1].max()

        # get x-axis and y-axis length
        xlen, ylen = xmax - xmin, ymax - ymin

        # extend x-axis and y-axis
        xmin, xmax = xmin - xlen * alpha, xmax + xlen * alpha
        ymin, ymax = ymin - ylen * alpha, ymax + ylen * alpha
        return pos_dict, ((xmin, xmax), (ymin, ymax))
    
    def hover_graph_info(self, graph):
        r"""Add hover information to graph

        Args
        ----
        graph : networkx.Graph
            The graph to visualize.

        """
        # Add node names
        for idx in graph.nodes:
            label = graph.nodes[idx]['label']
            _label = label[0].upper() + label[1:]
            graph.nodes[idx]['_label'] = _label
            _, _name = idx.split('_')
            graph.nodes[idx]['_name'] = _name
            graph.nodes[idx]['label_name'] = idx

        # Add edge names
        for idx1, idx2 in graph.edges:
            graph.edges[(idx1, idx2)]['label_name1'] = graph.nodes[idx1]['label_name']
            graph.edges[(idx1, idx2)]['label_name2'] = graph.nodes[idx2]['label_name']

    def get_node_source_id(self):
        r"""Get data source indices of all nodes in the graph"""
        # get requiring content
        self.int2node = self.renderer.node_renderer.data_source.data['index']
        self.node2int = {idx: i for i, idx in enumerate(self.int2node)}

    def get_edge_source_id(self):
        r"""Get data source indices of all edges in the graph"""
        # get requiring content
        self.int2edge = list(zip(
            self.renderer.edge_renderer.data_source.data['start'],
            self.renderer.edge_renderer.data_source.data['end']))
        self.edge2int = {(idx1, idx2): i for i, (idx1, idx2) in enumerate(self.int2edge)}

    def get_adj_source_id(self):
        r"""Get adjacent nodes info in source data"""
        # allocate link dictionary
        link_list = [set([]) for idx in self.graph.nodes]

        # build connections
        for (idx1, idx2) in self.graph.edges:
            e = self.edge2int[(idx1, idx2)]
            i1, i2 = self.node2int[idx1], self.node2int[idx2]
            link_list[i1].add((e, i2))
            link_list[i2].add((e, i1))
        self.adj_source = [list(itr) for itr in link_list]

    def sort_node_source(self):
        r"""Sort data source of node"""
        # sort values with string and int indices
        buffer = []
        label2int = dict(comic=0, hero=1)
        for idx in self.graph.nodes:
            i = self.node2int[idx]
            node_dict = self.graph.nodes[idx]
            lid = label2int[node_dict['label']]
            val = node_dict['sign'] if lid == 0 else node_dict['pr']
            buffer.append((i, idx, lid, val))
        buffer = sorted(buffer, key=lambda x: (x[-2], x[-1]))
        order = [itr[0] for itr in buffer]

        # rearange data source
        data_source = self.renderer.node_renderer.data_source
        for name in data_source.column_names:
            old_col = data_source.data[name]
            new_col = [None for itr in old_col]
            for new_i, old_i in enumerate(order):
                new_col[new_i] = old_col[old_i]
            data_source.data[name] = new_col

    def rescale(self, values, vmin, vmax, mask=None, mskval=0):
        r"""Linearly rescale values to given range

        Args
        ----
        values : numpy.ndarray
            Values to rescale.
        vmin : float
            Minimum scaled value.
        vmax : float
            Maximum scaled value.
        mask : numpy.ndarray
            Specify which value to jump.
        mskval : float
            Rescaled value for jumped values.

        Returns
        -------
        values : numpy.ndarray
            Rescaled values

        """
        # reset None mask
        if mask is None:
            mask = np.ones(shape=values.shape, dtype=bool)
        else:
            mask = mask.astype(bool)

        # get linear transformation
        if vmin != vmax:
            _values = values[mask]
            _bias = _values.min()
            _weight = (vmax - vmin) / (_values.max() - _values.min())
        else:
            pass

        # apply transformation
        if vmin != vmax:
            values = (values - _bias) * _weight + vmin
        else:
            values = np.zeros(shape=values.shape)
            values.fill(vmin)

        # reset masked values
        values[np.logical_not(mask)] = mskval
        return values

    def rescale_palette(self, values, palette, reverse=False, **kargs):
        r"""Linearly rescale values to given range of given palette

        Args
        ----
        values : numpy.ndarray
            Values to rescale.
        palette : bokeh.palette.*
            Palette.
        reverse : bool
            If rescale to palette by reversing order.

        Returns
        -------
        colors : [str, ...]
            A list of rescaled colors.

        """
        # get rescaled range in the palette
        if reverse:
            prange = self.rescale(-values, **kargs)
        else:
            prange = self.rescale(values, **kargs)

        # force range to be int
        prange = np.round(prange).astype(int)

        # get colors
        return [palette[itr] for itr in prange]

    def join_colors(self, color1, color2, mask1, mask2):
        r"""Join 2 color list together

        Args
        ----
        color1 : [str, ...]
            A list of colors.
        color2 : [str, ...]
            Another list of colors.
        mask1 : numpy.ndarray
            Mask of color1.
        mask2 : numpy.ndarray
            Mask of color2.

        """
        # create a new buffer
        colors = [None for i in range(max(len(color1), len(color2)))]

        # fill in colors based on masks
        for i in range(len(colors)):
            assert not (mask1[i] and mask2[i])
            if mask1[i]:
                colors[i] = color1[i]
            elif mask2[i]:
                colors[i] = color2[i]
            else:
                raise RuntimeError()
        return colors

    def set_nodes(self):
        r"""Set visualization properties for network nodes"""
        # get data source
        source = self.renderer.node_renderer.data_source
        node_data = source.to_df()
        label1    , label2     = 'comic'  , 'hero'
        size_col1 , size_col2  = 'sign'   , 'pr'
        color_col1, color_col2 = '#appear', '#know'
        palette1  , palette2   = BuGn9    , YlOrRd9

        # get array masks
        mask1 = (node_data['label'] == label1).values
        mask2 = (node_data['label'] == label2).values

        # --------------------------------------------------------------------------------

        # get node shape
        shape_vals = [None for i in range(len(node_data))]
        for i in range(len(node_data)):
            assert not (mask1[i] and mask2[i])
            if mask1[i]:
                shape_vals[i] = 'square'
            elif mask2[i]:
                shape_vals[i] = 'circle'
            else:
                raise RuntimeError()
        source.add(shape_vals, ':shape')

        # --------------------------------------------------------------------------------

        # get node size
        criterion1 = np.square(node_data[size_col1].values)
        criterion2 = np.square(node_data[size_col2].values)
        size_vals1 = self.rescale(criterion1, vmin=25, vmax=75, mask=mask1, mskval=0)
        size_vals2 = self.rescale(criterion2, vmin=25, vmax=75, mask=mask2, mskval=0)
        size_vals = size_vals1 + size_vals2
        source.add(size_vals, ':size')

        # --------------------------------------------------------------------------------

        # get node fill colors
        criterion1 = node_data[color_col1].values
        criterion2 = node_data[color_col2].values
        fill_color_vals1 = self.rescale_palette(
            criterion1, palette1, reverse=True, vmin=0, vmax=7, mask=mask1, mskval=8)
        fill_color_vals2 = self.rescale_palette(
            criterion2, palette2, reverse=True, vmin=0, vmax=7, mask=mask2, mskval=8)
        fill_color_vals = self.join_colors(fill_color_vals1, fill_color_vals2, mask1, mask2)
        source.add(fill_color_vals, ':fill_color')

        # get node fill alphas
        criterion1 = node_data[color_col1].values
        criterion2 = node_data[color_col2].values
        fill_alpha_vals1 = self.rescale(criterion1, vmin=0.85, vmax=0.85, mask=mask1, mskval=0)
        fill_alpha_vals2 = self.rescale(criterion2, vmin=0.85, vmax=0.85, mask=mask2, mskval=0)
        fill_alpha_vals = fill_alpha_vals1 + fill_alpha_vals2
        source.add(fill_alpha_vals, ':fill_alpha')

        # get node line colors
        criterion1 = node_data[color_col1].values
        criterion2 = node_data[color_col2].values
        line_color_vals1 = self.rescale_palette(
            criterion1, ['black'], reverse=False, vmin=0, vmax=0, mask=mask1, mskval=0)
        line_color_vals2 = self.rescale_palette(
            criterion2, ['black'], reverse=False, vmin=0, vmax=0, mask=mask2, mskval=0)
        line_color_vals = self.join_colors(line_color_vals1, line_color_vals2, mask1, mask2)
        source.add(line_color_vals, ':line_color')

        # get node line alphas
        criterion1 = node_data[color_col1].values
        criterion2 = node_data[color_col2].values
        line_alpha_vals1 = self.rescale(criterion1, vmin=0.85, vmax=0.85, mask=mask1, mskval=0)
        line_alpha_vals2 = self.rescale(criterion2, vmin=0.85, vmax=0.85, mask=mask2, mskval=0)
        line_alpha_vals = line_alpha_vals1 + line_alpha_vals2
        source.add(line_alpha_vals, ':line_alpha')

        # --------------------------------------------------------------------------------

        # get node fill alphas (non-selection)
        criterion1 = node_data[color_col1].values
        criterion2 = node_data[color_col2].values
        fill_alpha_vals1 = self.rescale(criterion1, vmin=0.01, vmax=0.01, mask=mask1, mskval=0)
        fill_alpha_vals2 = self.rescale(criterion2, vmin=0.01, vmax=0.01, mask=mask2, mskval=0)
        fill_alpha_vals = fill_alpha_vals1 + fill_alpha_vals2
        source.add(fill_alpha_vals, ':nosel:fill_alpha')

        # get node line alphas (non-selection)
        criterion1 = node_data[color_col1].values
        criterion2 = node_data[color_col2].values
        line_alpha_vals1 = self.rescale(criterion1, vmin=0.01, vmax=0.01, mask=mask1, mskval=0)
        line_alpha_vals2 = self.rescale(criterion2, vmin=0.01, vmax=0.01, mask=mask2, mskval=0)
        line_alpha_vals = line_alpha_vals1 + line_alpha_vals2
        source.add(line_alpha_vals, ':nosel:line_alpha')

        # --------------------------------------------------------------------------------

        # get node fill colors (hover)
        criterion1 = node_data[color_col1].values
        criterion2 = node_data[color_col2].values
        fill_color_vals1 = self.rescale_palette(
            criterion1, ['deepskyblue'], reverse=False, vmin=0, vmax=0, mask=mask1, mskval=0)
        fill_color_vals2 = self.rescale_palette(
            criterion2, ['deepskyblue'], reverse=False, vmin=0, vmax=0, mask=mask2, mskval=0)
        fill_color_vals = self.join_colors(fill_color_vals1, fill_color_vals2, mask1, mask2)
        source.add(fill_color_vals, ':hover:fill_color')

        # get node fill alphas
        criterion1 = node_data[color_col1].values
        criterion2 = node_data[color_col2].values
        fill_alpha_vals1 = self.rescale(criterion1, vmin=0.85, vmax=0.85, mask=mask1, mskval=0)
        fill_alpha_vals2 = self.rescale(criterion2, vmin=0.85, vmax=0.85, mask=mask2, mskval=0)
        fill_alpha_vals = fill_alpha_vals1 + fill_alpha_vals2
        source.add(fill_alpha_vals, ':hover:fill_alpha')

        # --------------------------------------------------------------------------------

        # reset node renderers
        self.renderer.node_renderer.glyph = Scatter(
            marker=':shape', size=':size', fill_color=':fill_color', fill_alpha=':fill_alpha',
            line_color=':line_color', line_alpha=':line_alpha')
        self.renderer.node_renderer.selection_glyph = Scatter(
            marker=':shape', size=':size', fill_color=':fill_color', fill_alpha=':fill_alpha',
            line_color=':line_color', line_alpha=':line_alpha')
        self.renderer.node_renderer.nonselection_glyph = Scatter(
            marker=':shape', size=':size', fill_color=':fill_color',
            fill_alpha=':nosel:fill_alpha', line_color=':line_color',
            line_alpha=':nosel:line_alpha')
        self.renderer.node_renderer.hover_glyph = Scatter(
            marker=':shape', size=':size', fill_color=':hover:fill_color',
            fill_alpha=':hover:fill_alpha', line_color=':line_color',
            line_alpha=':line_alpha')

        # reset node policies
        self.renderer.selection_policy = NodesAndLinkedEdges()
        self.renderer.inspection_policy = NodesAndLinkedEdges()

    def set_edges(self):
        r"""Set visualization properties for network edges"""
        # get data source
        source = self.renderer.edge_renderer.data_source
        edge_data = source.to_df()
        mode1     , mode2      = 'appear', 'know'
        size_col1 , size_col2  = 'cont'  , 'coop'
        color_col1, color_col2 = 'cont'  , 'coop'
        palette1  , palette2   = Greens9  , Reds9

        # get array masks
        mask1 = (edge_data['mode'] == mode1).values
        mask2 = (edge_data['mode'] == mode2).values

        # --------------------------------------------------------------------------------

        # get edge width
        criterion1 = np.square(edge_data[size_col1].values)
        criterion2 = np.square(edge_data[size_col2].values)
        width_vals1 = self.rescale(criterion1, vmin=2, vmax=16, mask=mask1, mskval=0)
        width_vals2 = self.rescale(criterion2, vmin=2, vmax=16, mask=mask2, mskval=0)
        width_vals = width_vals1 + width_vals2
        source.add(width_vals, ':width')

        # --------------------------------------------------------------------------------

        # get edge line colors
        criterion1 = edge_data[color_col1].values
        criterion2 = edge_data[color_col2].values
        line_color_vals1 = self.rescale_palette(
            criterion1, ['black'], reverse=False, vmin=0, vmax=0, mask=mask1, mskval=0)
        line_color_vals2 = self.rescale_palette(
            criterion2, ['black'], reverse=False, vmin=0, vmax=0, mask=mask2, mskval=0)
        line_color_vals = self.join_colors(line_color_vals1, line_color_vals2, mask1, mask2)
        source.add(line_color_vals, ':line_color')

        # get edge line alphas
        criterion1 = edge_data[color_col1].values
        criterion2 = edge_data[color_col2].values
        line_alpha_vals1 = self.rescale(criterion1, vmin=0.35, vmax=0.85, mask=mask1, mskval=0)
        line_alpha_vals2 = self.rescale(criterion2, vmin=0.35, vmax=0.85, mask=mask2, mskval=0)
        line_alpha_vals = line_alpha_vals1 + line_alpha_vals2
        source.add(line_alpha_vals, ':line_alpha')

        # --------------------------------------------------------------------------------

        # get edge line alphas (non-selection)
        criterion1 = edge_data[color_col1].values
        criterion2 = edge_data[color_col2].values
        line_alpha_vals1 = self.rescale(criterion1, vmin=0.01, vmax=0.01, mask=mask1, mskval=0)
        line_alpha_vals2 = self.rescale(criterion2, vmin=0.01, vmax=0.01, mask=mask2, mskval=0)
        line_alpha_vals = line_alpha_vals1 + line_alpha_vals2
        source.add(line_alpha_vals, ':nonsel:line_alpha')

        # --------------------------------------------------------------------------------

        # get edge line colors (hover)
        criterion1 = edge_data[color_col1].values
        criterion2 = edge_data[color_col2].values
        line_color_vals1 = self.rescale_palette(
            criterion1, ['cyan'], reverse=False, vmin=0, vmax=0, mask=mask1, mskval=0)
        line_color_vals2 = self.rescale_palette(
            criterion2, ['cyan'], reverse=False, vmin=0, vmax=0, mask=mask2, mskval=0)
        line_color_vals = self.join_colors(line_color_vals1, line_color_vals2, mask1, mask2)
        source.add(line_color_vals, ':hover:line_color')

        # get edge line alphas (hover)
        criterion1 = edge_data[color_col1].values
        criterion2 = edge_data[color_col2].values
        line_alpha_vals1 = self.rescale(criterion1, vmin=0.85, vmax=0.85, mask=mask1, mskval=0)
        line_alpha_vals2 = self.rescale(criterion2, vmin=0.85, vmax=0.85, mask=mask2, mskval=0)
        line_alpha_vals = line_alpha_vals1 + line_alpha_vals2
        source.add(line_alpha_vals, ':hover:line_alpha')

        # --------------------------------------------------------------------------------

        # reset node renderers
        self.renderer.edge_renderer.glyph = MultiLine(
            line_width=':width', line_color=':line_color', line_alpha=':line_alpha')
        self.renderer.edge_renderer.selection_glyph = MultiLine(
            line_width=':width', line_color=':line_color', line_alpha=':line_alpha')
        self.renderer.edge_renderer.nonselection_glyph = MultiLine(
            line_width=':width', line_color=':line_color', line_alpha=':nonsel:line_alpha')
        self.renderer.edge_renderer.hover_glyph = MultiLine(
            line_width=':width', line_color=':hover:line_color',
            line_alpha=':hover:line_alpha')

    def set_tools(self):
        r"""Set tools"""
        # create graph independent tools
        pan = PanTool()
        wheel_zoom = WheelZoomTool()
        hover = HoverTool(
            tooltips=[('Label', "@{_label}"), ('Name', "@{_name}")], point_policy='follow_mouse')
        tap = TapTool()

        # add tools to plot
        self.plot.add_tools(pan, wheel_zoom, hover, tap)

        # force wheel zoom tool to be active
        self.plot.toolbar.active_scroll = wheel_zoom

    def from_graph(self, graph):
        r"""Get visualization plot from graph

        Args
        ----
        graph : networkx.Graph
            The graph to visualize.

        """
        # update additional info
        self.hover_graph_info(graph)

        # get position related data
        pos_dict, ((xmin, xmax), (ymin, ymax)) = self.get_graph_pos(graph)
    
        # get basic graph renderers
        self.graph = graph
        self.renderer = from_networkx(self.graph, pos_dict)

        # refresh node data for inspection and update convenience
        self.get_node_source_id()
        self.get_edge_source_id()
        self.sort_node_source()

        self.get_node_source_id()
        self.get_edge_source_id()
        self.get_adj_source_id()

        # set graph renderer details
        self.set_nodes()
        self.set_edges()

        # initialize plot
        self.plot = Plot(
            plot_width=self.PLOT_W, plot_height=self.PLOT_H, x_range=Range1d(xmin, xmax),
            y_range=Range1d(ymin, ymax))

        # set tools
        self.set_tools()

        # set graph visualization
        self.plot.renderers.append(self.renderer)
