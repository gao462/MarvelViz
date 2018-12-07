The Marvel Universe Social Network Visualization
===

[TOC]

## Who

I work on this project individually.

- Jianfei Gao
- 0029986102

## Demo

Clone from [my repository](https://github.com/gao462/MarvelViz), and run `bokeh serve --show draw.py`. The `bokeh` library I used to build this project is version 1.0.2 from conda-forge channel.

## What - Dataset

Raw dataset comes from [The Marvel Universe Social Network](https://www.kaggle.com/csanhueza/the-marvel-universe-social-network) on Kaggle. I also include the raw dataset in my repository.

Only two files are used in this visualization project:

1. edges.csv / appear.csv

   A list of the relationships that a hero appears in a volume of comic.

2. hero-network.csv / know.csv

   A list of the relationships that a hero knows another hero.

**The dataset is equivalent to two undirected graphs with sharing nodes (heroes).** The appearance relationship graph is a bipartite graph, and the knowing relationship graph is a common network.

## Why - Target

1. Who are important heroes in the Marvel Universe?

2. Which comics are significance events in the Marvel Universe?

3. For a hero, I want to know his/her social statistics and detail information.

## How - Preprocessing

The first thing is that the raw data is noisy. For example, it includes two hero nodes "IRONMAN/TONY STARK" and "IRONMAN/TONY STARK " (with a tail space) which indeed are the same thing. So, the first thing I did is preprocessing all names to merge those falsely duplicated nodes by `fuzzywuzzy` library.

The merging result is also provided in `data/_name_fath.p` in my repository.

To visualize the importance of heroes and significance of comics, we need manually add more information to the dataset. Here is the information I added.

| Importance      | Significance                         |
| --------------- | ------------------------------------ |
| Node Degree     | Node Degree                          |
| PageRank Result | Sum of PageRank Results of Neighbors |

To get PageRank result, I use `networkx` library.

I also want to quantify appearance and knowing relationships, so that I can have a better understanding of social statistics of heroes.

- Appearance Relationship

  I use the inverse of the degree of connecting comic nodes as the weight of this relationship. Thus, higher weight means the comic being more concentrated on the hero.

- Knowing Relationship

  I use the number of sharing comics of two connecting heroes as the weight of this relationship. Thus, the higher weight means longer cooperation experience, which can be regarded as the strength of their relationship.

## How - Visualize

- [x] Visualize hero node (circle, warm color) and comic node (square, cold color) in different shapes and palettes.
- [x] Visualize the relationship between nodes.
- [x] Visualize importance and significance of node degree by different node colors. Because it is discrete value.
- [x] Visualize importance and significance of PageRank by different node size. Because it is continuous value.
- [x] Visualize weight of relationships by edge width and edge color saturation. Because it is continuous value.
- [x] Use sliders to control the size of nodes and the width of edges.
- [x] Use range sliders to control the percentage of visualizing nodes and edges. For example, it can exclude the least important 40% nodes and the most important 10% nodes.
- [x] Provide legends for users to understand the values corresponding to visualizing geometries
- [x] On clicking a node, only highlight all its neighbors with edges and itself.
- [x] On clicking a node, the detail name and values of that node will be shown on the left side.
- [x] On clicking a node, it will automatically do google search and put the first image result on the left side to help users know who or which event they are investigating.
- [x] On clicking a node, its neighbor statistics is shown in pie and bar charts
  - The hero node degree importance distribution of its neighbor heroes
  - The comic node degree significance distribution of its neighbor comics
  - The appearance weight distribution of its connecting relationships
  - The knowing weight distribution of its connecting relationships
- [x] On clicking a node, its rank of importance and significance will he shown in the global distribution charts on the left side.
- [x] On clicking a bar of neighbor statistics visualization, it will only highlight selection nodes and its neighbors and relationships that falls into the selected bar.
- [x] A text input is provided to search a specific node by name. For example, type in "IRONMAN" and search, and it will automatically select "IRONMAN/TONY STARK".

## Deprecated

- [ ] If we select two heroes, show the shortest communication path on social network for them to get in touch. **Deprecated Reason: `bokeh` for now does not support multi mouse selection callbacks.**
- [ ] If multi nodes are selected, show sharing neighbors of them in different highlighting schema. **Deprecated Reason: `bokeh` for now does not support multi mouse selection callbacks.**