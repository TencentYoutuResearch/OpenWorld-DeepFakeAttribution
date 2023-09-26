#!/usr/bin/env python
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from loguru import logger

from openTSNE import TSNE


MACOSKO_COLORS = {
    "Amacrine cells": "#A5C93D",
    "Astrocytes": "#8B006B",
    "Bipolar cells": "#2000D7",
    "Cones": "#538CBA",
    "Fibroblasts": "#8B006B",
    "Horizontal cells": "#B33B19",
    "Microglia": "#8B006B",
    "Muller glia": "#8B006B",
    "Pericytes": "#8B006B",
    "Retinal ganglion cells": "#C38A1F",
    "Rods": "#538CBA",
    "Vascular endothelium": "#8B006B",
}
ZEISEL_COLORS = {
    "Astroependymal cells": "#d7abd4",
    "Cerebellum neurons": "#2d74bf",
    "Cholinergic, monoaminergic and peptidergic neurons": "#9e3d1b",
    "Di- and mesencephalon neurons": "#3b1b59",
    "Enteric neurons": "#1b5d2f",
    "Hindbrain neurons": "#51bc4c",
    "Immature neural": "#ffcb9a",
    "Immune cells": "#768281",
    "Neural crest-like glia": "#a0daaa",
    "Oligodendrocytes": "#8c7d2b",
    "Peripheral sensory neurons": "#98cc41",
    "Spinal cord neurons": "#c52d94",
    "Sympathetic neurons": "#11337d",
    "Telencephalon interneurons": "#ff9f2b",
    "Telencephalon projecting neurons": "#fea7c1",
    "Vascular cells": "#3d672d",
}
MOUSE_10X_COLORS = {
    0: "#FFFF00",
    1: "#1CE6FF",
    2: "#FF34FF",
    3: "#FF4A46",
    4: "#008941",
    5: "#006FA6",
    6: "#A30059",
    7: "#FFDBE5",
    8: "#7A4900",
    9: "#0000A6",
    10: "#63FFAC",
    11: "#B79762",
    12: "#004D43",
    13: "#8FB0FF",
    14: "#997D87",
    15: "#5A0007",
    16: "#809693",
    17: "#FEFFE6",
    18: "#1B4400",
    19: "#4FC601",
    20: "#3B5DFF",
    21: "#4A3B53",
    22: "#FF2F80",
    23: "#61615A",
    24: "#BA0900",
    25: "#6B7900",
    26: "#00C2A0",
    27: "#FFAA92",
    28: "#FF90C9",
    29: "#B903AA",
    30: "#D16100",
    31: "#DDEFFF",
    32: "#000035",
    33: "#7B4F4B",
    34: "#A1C299",
    35: "#300018",
    36: "#0AA6D8",
    37: "#013349",
    38: "#00846F",
}


def plot(
    x,
    y,
    ax=None,
    title=None,
    draw_legend=True,
    draw_centers=False,
    draw_cluster_labels=False,
    colors=None,
    legend_kwargs=None,
    label_order=None,
    **kwargs
):
    import matplotlib

    if ax is None:
        _, ax = matplotlib.pyplot.subplots(figsize=(12, 12))

    if title is not None:
        ax.set_title(title)

    plot_params = {"alpha": kwargs.get("alpha", 0.8), "s": kwargs.get("s", 1)}

    # Create main plot
    if label_order is not None:
        assert all(np.isin(np.unique(y), label_order))
        classes = [l for l in label_order if l in np.unique(y)]
    else:
        classes = np.unique(y)
    if colors is None:
        default_colors = matplotlib.rcParams["axes.prop_cycle"]
        colors = {k: v["color"] for k, v in zip(classes, default_colors())}

    point_colors = list(map(colors.get, y))

    ax.scatter(x[:, 0], x[:, 1], c=point_colors,
               rasterized=True, **plot_params)

    # Plot mediods
    if draw_centers:
        centers = []
        for yi in classes:
            mask = yi == y
            centers.append(np.median(x[mask, :2], axis=0))
        centers = np.array(centers)

        center_colors = list(map(colors.get, classes))
        ax.scatter(
            centers[:, 0], centers[:, 1], c=center_colors, s=48, alpha=1, edgecolor="k"
        )

        # Draw mediod labels
        if draw_cluster_labels:
            for idx, label in enumerate(classes):
                ax.text(
                    centers[idx, 0],
                    centers[idx, 1] + 2.2,
                    label,
                    fontsize=kwargs.get("fontsize", 6),
                    horizontalalignment="center",
                )

    # Hide ticks and axis
    # ax.set_xticks([]), ax.set_yticks([]), ax.axis("off")

    if draw_legend:
        legend_handles = [
            matplotlib.lines.Line2D(
                [],
                [],
                marker="s",
                color="w",
                markerfacecolor=colors[yi],
                ms=10,
                alpha=1,
                linewidth=0,
                label=yi,
                markeredgecolor="k",
            )
            for yi in classes
        ]
        legend_kwargs_ = dict(
            loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, )
        if legend_kwargs is not None:
            legend_kwargs_.update(legend_kwargs)
        ax.legend(handles=legend_handles, **legend_kwargs_)

    canvas = FigureCanvasAgg(plt.gcf())
    canvas.draw()
    return np.array(canvas.buffer_rgba())


def feature_tsne(x, y):
    logger.info(f'Info of data: {x.shape}, {y.shape}')
    logger.info('TSNE working...')
    x_tsne = TSNE(n_components=2, n_jobs=8).fit(x)
    return plot(x_tsne, y, draw_centers=True, draw_cluster_labels=True, colors=MOUSE_10X_COLORS)


def visualize(tags, labels, feats):
    logger.info(f'Shape of feature and labels: {feats.shape}, {labels.shape}')

    num_known = np.max(labels[tags == 1], axis=0) + 1
    known_mask = labels < num_known
    novel_mask = ~known_mask

    # Known Attack
    known_idxs = np.where(known_mask)[0]
    known_feats = feats[known_idxs]
    known_labels = labels[known_idxs]
    fig1 = feature_tsne(known_feats, known_labels)

    # Novel Attack
    novel_idxs = np.where(novel_mask)[0]
    novel_feats = feats[novel_idxs]
    novel_labels = labels[novel_idxs]
    fig2 = feature_tsne(novel_feats, novel_labels)

    # All Attack
    fig3 = feature_tsne(feats, labels)

    return fig1, fig2, fig3
