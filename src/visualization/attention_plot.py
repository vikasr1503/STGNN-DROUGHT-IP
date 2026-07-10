"""
attention_plot.py
=================
Visualises GAT attention coefficients over the sub-basin graph.
Shows which upstream sub-basins a given node attends to most strongly.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import torch.nn as nn

RESULTS = Path("results/figures")
RESULTS.mkdir(parents=True, exist_ok=True)


def extract_attention_weights(model: nn.Module,
                               batch: dict,
                               edge_index: torch.Tensor,
                               device: torch.device) -> np.ndarray:
    """
    Forward pass through GATGRUModel and extract attention coefficients
    from the first GAT layer at the last timestep.

    Returns attention array of shape [num_edges, num_heads].
    """
    from torch_geometric.nn import GATv2Conv

    model.eval()
    attention_weights = {}

    def hook_fn(module, input, output):
        # GATv2Conv returns (out, attention) when return_attention_weights=True
        if isinstance(output, tuple):
            attention_weights["attn"] = output[1][1].detach().cpu().numpy()

    # Register hook on first GAT layer
    first_gat = None
    for module in model.modules():
        if isinstance(module, GATv2Conv):
            first_gat = module
            break

    if first_gat is None:
        raise ValueError("No GATv2Conv layer found in model.")

    handle = first_gat.register_forward_hook(hook_fn)

    with torch.no_grad():
        X  = batch["X"].to(device)
        ew = batch.get("edge_weight")
        if ew is not None:
            ew = ew.to(device)
        model(X, edge_index.to(device), edge_weight=ew)

    handle.remove()
    return attention_weights.get("attn", np.array([]))


def plot_attention_heatmap(attention: np.ndarray,
                           edges: pd.DataFrame,
                           nodes: pd.DataFrame,
                           target_node: int = 0,
                           head: int = 0,
                           save: bool = True) -> None:
    """
    Plot attention weights from neighbours to a target sub-basin node.

    attention: [num_edges, num_heads]
    edges:     edge DataFrame with from_subbasin, to_subbasin, src_idx, dst_idx
    target_node: node_idx of the sub-basin to visualise
    head:        which attention head to plot
    """
    # Filter edges arriving at target_node
    incoming = edges[edges["dst_idx"] == target_node].copy()
    if incoming.empty:
        print(f"No incoming edges to node {target_node}.")
        return

    incoming["attention"] = attention[incoming.index % len(attention), head]
    incoming = incoming.sort_values("attention", ascending=False)

    src_names = incoming["from_subbasin"].astype(str).values
    attn_vals = incoming["attention"].values

    fig, ax = plt.subplots(figsize=(10, max(4, len(src_names) * 0.35)))
    colors = cm.YlOrRd(attn_vals / (attn_vals.max() + 1e-8))
    ax.barh(src_names[::-1], attn_vals[::-1], color=colors[::-1])
    ax.set_xlabel("Attention Weight")
    ax.set_ylabel("Source Sub-basin")
    ax.set_title(f"GAT Attention (Head {head}) → Sub-basin {target_node}")
    plt.tight_layout()

    if save:
        path = RESULTS / f"attention_node{target_node}_head{head}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {path}")
    else:
        plt.show()


def plot_attention_graph(attention: np.ndarray,
                         edges: pd.DataFrame,
                         nodes: pd.DataFrame,
                         head: int = 0,
                         top_k: int = 50,
                         save: bool = True) -> None:
    """
    Spatial map of top-k attention edges coloured by weight.
    """
    import networkx as nx

    attn_vals = attention[:, head] if attention.ndim == 2 else attention
    edges = edges.copy()
    edges["attn"] = attn_vals[:len(edges)]
    top_edges = edges.nlargest(top_k, "attn")

    G = nx.DiGraph()
    for _, row in nodes.iterrows():
        G.add_node(row["subbasin_id"],
                   pos=(row.get("centroid_lon", 0),
                        row.get("centroid_lat", 0)))

    edge_colors = []
    for _, row in top_edges.iterrows():
        G.add_edge(int(row["from_subbasin"]), int(row["to_subbasin"]),
                   weight=float(row["attn"]))
        edge_colors.append(float(row["attn"]))

    pos = nx.get_node_attributes(G, "pos")
    fig, ax = plt.subplots(figsize=(12, 10))

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=30,
                           node_color="steelblue", alpha=0.7)
    ec = nx.draw_networkx_edges(G, pos, ax=ax,
                                edge_color=edge_colors,
                                edge_cmap=cm.YlOrRd,
                                arrows=True, arrowsize=8,
                                width=1.5, alpha=0.8,
                                connectionstyle="arc3,rad=0.1")
    ax.set_title(f"Top-{top_k} Attention Edges (Head {head})")
    ax.axis("off")
    plt.tight_layout()

    if save:
        path = RESULTS / f"attention_graph_head{head}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
