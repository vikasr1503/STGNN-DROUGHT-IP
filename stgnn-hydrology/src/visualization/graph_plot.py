"""
graph_plot.py
=============
Visualise the sub-basin graph structure over the Krishna River Basin.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

RESULTS = Path("results/figures")
RESULTS.mkdir(parents=True, exist_ok=True)


def plot_graph(nodes: pd.DataFrame, edges: pd.DataFrame,
               title: str = "Krishna Basin Sub-basin Graph",
               save: bool = True) -> None:
    """
    Plot the directed sub-basin connectivity graph.
    Nodes sized by area, coloured by elevation.
    """
    G = nx.DiGraph()
    for _, n in nodes.iterrows():
        G.add_node(n["subbasin_id"],
                   pos=(n.get("centroid_lon", 0), n.get("centroid_lat", 0)),
                   area=n.get("area_km2", 1),
                   elev=n.get("elevation_m", 0))

    for _, e in edges.iterrows():
        G.add_edge(int(e["from_subbasin"]), int(e["to_subbasin"]))

    pos    = nx.get_node_attributes(G, "pos")
    areas  = np.array([G.nodes[n]["area"] for n in G.nodes])
    elevs  = np.array([G.nodes[n]["elev"] for n in G.nodes])

    fig, ax = plt.subplots(figsize=(12, 10))
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="steelblue",
                           arrows=True, arrowsize=10, alpha=0.5,
                           connectionstyle="arc3,rad=0.1")
    sc = nx.draw_networkx_nodes(G, pos, ax=ax,
                                node_size=np.clip(areas * 0.05, 20, 200),
                                node_color=elevs, cmap="terrain", alpha=0.9)
    plt.colorbar(sc, ax=ax, label="Elevation (m)")
    ax.set_title(title, fontsize=14)
    ax.axis("off")

    if save:
        path = RESULTS / "subbasin_graph.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {path}")
    else:
        plt.show()


def plot_drought_map(nodes: pd.DataFrame,
                     sri_values: np.ndarray,
                     date: str,
                     save: bool = True) -> None:
    """
    Choropleth-style map of SRI values across sub-basins for one timestep.
    sri_values: array of length N
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    lons = nodes["centroid_lon"].values
    lats = nodes["centroid_lat"].values

    sc = ax.scatter(lons, lats, c=sri_values, cmap="RdYlBu",
                    vmin=-3, vmax=3, s=80, edgecolors="k", linewidths=0.3)
    plt.colorbar(sc, ax=ax, label="SRI")
    ax.set_title(f"SRI Spatial Distribution — {date}", fontsize=13)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    if save:
        path = RESULTS / f"sri_map_{date.replace('-', '')}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
