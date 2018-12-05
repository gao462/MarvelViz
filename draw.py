from viz.source import RawGraph, ProcessedBiGraph
from viz.plot import GraphViz
from viz.interact import GraphWidget, BiGraphWidget
from bokeh.plotting import curdoc


G = RawGraph()
G.preprocess_hero_name('data')
G.from_raw('data')
G.save('data/raw_graph.p')
 
G = RawGraph()
G.load('data/raw_graph.p')
G.pagerank()
G.save('data/raw_graph.p')
 
G = RawGraph()
G.load('data/raw_graph.p')
G.weight()
G.save('data/raw_graph.p')

G = RawGraph()
G.load('data/raw_graph.p')
BG = ProcessedBiGraph()
BG.from_raw_graph(G)
BG.save('data/processed_bigraph.p')

BG = ProcessedBiGraph()
BG.load('data/processed_bigraph.p')
BG.layout(BG.minor)
BG.save('data/processed_bigraph.p')

BG = ProcessedBiGraph()
BG.load('data/processed_bigraph.p')
BG.layout(BG.major)
BG.save('data/processed_bigraph.p')

BG = ProcessedBiGraph()
BG.load('data/processed_bigraph.p')
BG.major = BG.truncate(BG.major, num_heros=100, num_hero_comics=3)
BG.minor = BG.truncate(BG.minor, num_heros=100, num_hero_comics=3)
BG.layout(BG.major)
BG.layout(BG.minor)
BG.save('data/viz.p')

# BG = ProcessedBiGraph()
# BG.load('data/viz.p')
# viz = BiGraphWidget(GraphWidget(GraphViz(BG.major)), GraphWidget(GraphViz(BG.minor)))
# curdoc().add_root(viz.layout)