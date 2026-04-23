from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx

# Node colours by stage
COLOURS = {
    "topk":     "#2ECC71",  # green  — final top-k
    "expanded": "#E67E22",  # orange — added by AST expansion
    "stage1":   "#3498DB",  # blue   — dual-encoder candidates not in top-k
    "context":  "#BDC3C7",  # grey   — neighbours shown for structure only
}


def visualize_search(
    graph: nx.DiGraph,
    stage1_ids: set[str],
    expanded_ids: set[str],
    topk_ids: set[str],
    out_path: Path,
    context_hops: int = 1,
) -> None:
    """
    Draw a subgraph highlighting nodes by pipeline stage and save to out_path.

    Colour priority (highest wins): top-k > expanded > stage-1 > context.

    Parameters
    ----------
    graph        : full call graph from SnippetIndex
    stage1_ids   : snippet IDs returned by the dual encoder (step 1)
    expanded_ids : snippet IDs *added* by AST expansion (step 2 additions only)
    topk_ids     : final top-k snippet IDs (step 3)
    out_path     : where to write the PNG
    context_hops : how many extra hops to include around relevant nodes for
                   structural context (default 1)
    """
    relevant = stage1_ids | expanded_ids | topk_ids

    # collect context nodes (neighbours of relevant nodes up to context_hops)
    context_ids: set[str] = set()
    frontier = relevant & set(graph.nodes())
    for _ in range(context_hops):
        next_frontier: set[str] = set()
        for nid in frontier:
            next_frontier.update(graph.predecessors(nid))
            next_frontier.update(graph.successors(nid))
        new = (next_frontier - relevant - context_ids) & set(graph.nodes())
        context_ids.update(new)
        frontier = new

    all_nodes = (relevant | context_ids) & set(graph.nodes())
    if not all_nodes:
        return

    sub = graph.subgraph(all_nodes).copy()

    def node_colour(nid: str) -> str:
        if nid in topk_ids:
            return COLOURS["topk"]
        if nid in expanded_ids:
            return COLOURS["expanded"]
        if nid in stage1_ids:
            return COLOURS["stage1"]
        return COLOURS["context"]

    def node_size(nid: str) -> int:
        if nid in topk_ids:
            return 900
        if nid in expanded_ids or nid in stage1_ids:
            return 500
        return 200

    def short_label(nid: str) -> str:
        # qualified_name is the third :: segment
        parts = nid.split("::")
        return parts[2] if len(parts) >= 3 else nid

    node_colours = [node_colour(n) for n in sub.nodes()]
    node_sizes   = [node_size(n)   for n in sub.nodes()]
    labels       = {n: short_label(n) for n in sub.nodes()}

    n = sub.number_of_nodes()
    figsize = max(10, n // 3)
    fig, ax = plt.subplots(figsize=(figsize, figsize))

    pos = nx.spring_layout(sub, seed=42, k=3.0 / max(n ** 0.5, 1))

    nx.draw_networkx_nodes(sub, pos, node_color=node_colours,
                           node_size=node_sizes, alpha=0.9, ax=ax)
    nx.draw_networkx_edges(sub, pos, edge_color="#7F8C8D", arrows=True,
                           arrowsize=12, width=0.8, ax=ax,
                           connectionstyle="arc3,rad=0.08")
    nx.draw_networkx_labels(sub, pos, labels=labels, font_size=7, ax=ax)

    legend_handles = [
        mpatches.Patch(color=COLOURS["topk"],     label=f"Top-k results ({len(topk_ids)})"),
        mpatches.Patch(color=COLOURS["expanded"],  label=f"AST expansion ({len(expanded_ids)})"),
        mpatches.Patch(color=COLOURS["stage1"],    label=f"Dual-encoder ({len(stage1_ids - topk_ids)})"),
        mpatches.Patch(color=COLOURS["context"],   label=f"Context neighbours ({len(context_ids)})"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=9)
    ax.set_title(f"Search graph — {n} nodes", fontsize=11)
    ax.axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[visualize] saved → {out_path}")
