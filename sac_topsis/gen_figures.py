#!/usr/bin/env python3
"""
gen_figures.py — Gera figuras de publicação para o artigo SAC-TOPSIS.

Figuras geradas em images/:
  fig_01_architecture.pdf       — Fluxo do SAC-TOPSIS (diagrama em blocos)
  fig_02_performance.pdf        — Comparativo principal (deadline + SWQ + energy)
  fig_03_generalization.pdf     — Generalização zero-shot por densidade vehicular
  fig_04_modes.pdf              — Distribuição dos modos de execução
  fig_05_training.pdf           — Curvas de convergência SAC-TOPSIS vs SAC-Base
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
import re

warnings.filterwarnings("ignore")

# ── Caminhos ──────────────────────────────────────────────────────────────────
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR   = "/run/media/sergiosvieira/TOSHIBA EXT/Documentos/2026.1/vec_sim/test_results"
MODELS_DIR    = "/run/media/sergiosvieira/TOSHIBA EXT/Documentos/2026.1/vec_sim/models"
OUT_DIR       = os.path.join(SCRIPT_DIR, "images")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Estilo publicação ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "font.size":          9,
    "axes.titlesize":     9,
    "axes.labelsize":     9,
    "legend.fontsize":    7.5,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "figure.dpi":         200,
    "savefig.dpi":        300,
    "axes.grid":          True,
    "grid.alpha":         0.25,
    "grid.linewidth":     0.5,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "lines.linewidth":    1.8,
    "patch.linewidth":    0.6,
})

# ── Paleta consistente ────────────────────────────────────────────────────────
PAL = {
    "sac-topsis":   "#0D47A1",   # azul escuro
    "sac-base":     "#90CAF9",   # azul claro
    "td3-topsis":   "#1565C0",   # azul médio
    "ppo-topsis":   "#E65100",   # laranja escuro
    "ppo-base":     "#FFCC80",   # laranja claro
    "local":        "#546E7A",   # cinza azulado
    "greedy":       "#8D6E63",   # marrom
    "greedy-fair":  "#BCAAA4",   # marrom claro
    "greedy-unfair":"#4E342E",   # marrom escuro (oracle)
    "random":       "#B0BEC5",   # cinza
}

LABELS = {
    "sac-topsis":   "SAC-TOPSIS\n(proposed)",
    "sac-base":     "SAC-Base",
    "td3-topsis":   "TD3-TOPSIS",
    "ppo-topsis":   "PPO-TOPSIS",
    "ppo-base":     "PPO-Base",
    "local":        "Always-Local",
    "greedy":       "Greedy",
    "greedy-fair":  "Greedy-Fair",
    "greedy-unfair":"Greedy-Oracle",
    "random":       "Random",
}

def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, bbox_inches="tight", format="pdf")
    plt.close(fig)
    print(f"  [OK] {name}")

# ── Carrega dados ─────────────────────────────────────────────────────────────
def load_summary():
    path = os.path.join(RESULTS_DIR, "summary.csv")
    df = pd.read_csv(path)
    df["tps"] = df["cenario"].str.extract(r"_(\d+)tps_").astype(int)
    df = df[df["tps"] != 45].reset_index(drop=True)
    return df

def smooth(s, w=20):
    return s.rolling(window=w, min_periods=1, center=True).mean()

# ─────────────────────────────────────────────────────────────────────────────
# Fig 1 — Diagrama de arquitetura SAC-TOPSIS (matplotlib blocos)
# ─────────────────────────────────────────────────────────────────────────────
def fig_architecture():
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, w, h, color, label, fontsize=8, bold=False):
        rect = mpatches.FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.1", linewidth=1.0,
            edgecolor="#333333", facecolor=color, zorder=3
        )
        ax.add_patch(rect)
        weight = "bold" if bold else "normal"
        ax.text(x, y, label, ha="center", va="center",
                fontsize=fontsize, weight=weight, zorder=4, wrap=True)

    def arrow(x1, y1, x2, y2, label="", color="#333"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.2), zorder=2)
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx, my+0.15, label, ha="center", fontsize=6.5, color=color)

    # Entradas
    box(1.5, 5.0, 2.4, 0.8, "#E3F2FD", "Task $\\tau_j$\n($C_j$, $D_j$, $\\mathbf{x}_j$)")
    box(1.5, 3.5, 2.4, 0.8, "#E3F2FD", "Env. Observations\n(RSU, V2V, queues)")

    # TOPSIS block
    box(5.0, 4.25, 2.8, 2.4, "#FFF9C4",
        "TOPSIS Module\n\n"
        "$C_1$=JCT (0.4)\n$C_2$=Energy (0.3)\n$C_3$=Distance (0.3)\n\n"
        "Manhattan $L_1$\nAll-Minimization",
        fontsize=7.5)

    # State vector
    box(8.2, 4.25, 2.4, 2.4, "#E8F5E9",
        "6D State $\\mathbf{s}$\n\n"
        "$s_1$: Task complexity\n"
        "$s_2$: $\\delta_{\\mathrm{obj}}$ offload gain\n"
        "$s_3$: $t_{\\mathrm{loc}}/D_j$\n"
        "$s_4$: $t_{\\mathrm{r1}}/D_j$\n"
        "$s_5$: RSU queue\n"
        "$s_6$: Local queue",
        fontsize=7.0)

    # SAC agent
    box(11.5, 4.25, 2.0, 2.4, "#FCE4EC",
        "SAC Agent\n\n"
        "Max-Entropy\nPolicy\n\n"
        "$a \\in \\{0,1\\}$\nLocal / Offload",
        fontsize=7.5, bold=False)

    # Execute block
    box(11.5, 1.4, 2.0, 1.2, "#F3E5F5",
        "Execute\nRank-1\nCandidate",
        fontsize=7.5)

    # Rank-1 annotation
    box(8.2, 1.4, 2.4, 1.2, "#ECEFF1",
        "Rank-1: Best offload\n(RSU or V2V)",
        fontsize=7.5)

    # Reward
    box(7.0, 0.7, 4.0, 0.75, "#FBE9E7",
        "Deadline-Aware Reward: $r \\in [-2, +1]$  (SAC update)",
        fontsize=7.5)

    # Arrows
    arrow(2.7, 5.0, 3.6, 4.6, "")
    arrow(2.7, 3.5, 3.6, 3.9, "")
    arrow(6.4, 4.25, 7.0, 4.25, "rank-1")
    arrow(9.4, 4.25, 10.5, 4.25, "$\\mathbf{s}$")
    arrow(12.5, 3.0, 12.5, 2.0, "$a=1$")
    arrow(9.4, 1.4, 10.5, 1.4, "")
    arrow(12.5, 0.8, 9.0, 0.7, "")
    # action=0 arrow
    ax.annotate("", xy=(1.5, 1.4), xytext=(10.5, 1.4),
                arrowprops=dict(arrowstyle="->", color="#546E7A", lw=1.0,
                                connectionstyle="arc3,rad=0.0"), zorder=2)
    ax.text(6.0, 1.7, "$a=0$: local exec.", ha="center", fontsize=6.5, color="#546E7A")

    ax.set_title(
        "SAC-TOPSIS Architecture: TOPSIS State Compression + Maximum-Entropy Policy",
        fontsize=9, pad=6
    )
    save(fig, "fig_01_architecture.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 2 — Comparativo principal: 3 métricas, políticas selecionadas
# ─────────────────────────────────────────────────────────────────────────────
def fig_performance(df):
    POLS = ["sac-topsis", "td3-topsis", "ppo-topsis", "ppo-base", "local",
            "greedy-unfair", "greedy", "greedy-fair", "random", "sac-base"]
    agg = df.groupby("politica").agg(
        deadline=("taxa_deadline", "mean"),
        swq=("swq", "mean"),
        jct=("jct_medio_s", "mean"),
        energy=("energia_total_J", "mean"),
    ).reset_index()
    agg = agg[agg["politica"].isin(POLS)].copy()
    agg["politica"] = pd.Categorical(agg["politica"], categories=POLS, ordered=True)
    agg = agg.sort_values("politica").reset_index(drop=True)
    labels = [LABELS.get(p, p).replace("\n", " ") for p in agg["politica"]]
    colors = [PAL.get(p, "#999") for p in agg["politica"]]

    fig = plt.figure(figsize=(7.16, 5.8))
    gs = GridSpec(2, 2, figure=fig, hspace=0.52, wspace=0.38)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    def hbar(ax, values, title, xlabel, highlight_max=True, pct=False, fmt="{:.2f}"):
        idxs = np.arange(len(values))
        bars = ax.barh(idxs, values, color=colors, edgecolor="white", height=0.7)
        best = values.max() if highlight_max else values.min()
        for bar, val in zip(bars, values):
            if abs(val - best) < 1e-4:
                bar.set_edgecolor("#FFD700")
                bar.set_linewidth(2.0)
            suffix = "%" if pct else ""
            ax.text(val + 0.005 * values.max(), bar.get_y() + bar.get_height()/2,
                    fmt.format(val) + suffix,
                    va="center", fontsize=6.2)
        ax.set_yticks(idxs)
        ax.set_yticklabels(labels, fontsize=7.0)
        ax.set_title(title, fontsize=8.5, pad=4)
        ax.set_xlabel(xlabel, fontsize=7.5)
        ax.set_xlim(0, values.max() * 1.22)
        ax.invert_yaxis()

    hbar(ax1, agg["deadline"], "(a) Deadline Compliance Rate",
         "Mean rate (%) — higher is better", highlight_max=True, pct=True, fmt="{:.1f}")
    hbar(ax2, agg["swq"],
         r"(b) SWQ = $\hat\tau \cdot \overline{\mathrm{slack}}$",
         "SWQ index — higher is better", fmt="{:.4f}")
    hbar(ax3, agg["jct"],
         "(c) Mean Job Completion Time",
         "Mean JCT (s) — lower is better", highlight_max=False, fmt="{:.2f}")
    hbar(ax4, agg["energy"],
         "(d) Total Energy Consumed",
         "Mean energy (J) — lower is better", highlight_max=False, fmt="{:.0f}")

    # Patch legend
    h1 = mpatches.Patch(color=PAL["sac-topsis"], label="SAC-TOPSIS (proposed)")
    h2 = mpatches.Patch(color=PAL["sac-base"],   label="DRL Baselines")
    h3 = mpatches.Patch(color=PAL["greedy"],      label="Heuristics")
    fig.legend(handles=[h1, h2, h3], loc="upper center", ncol=3,
               fontsize=7.5, bbox_to_anchor=(0.5, 1.01), frameon=True)

    save(fig, "fig_02_performance.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 3 — Generalização zero-shot por densidade vehicular
# ─────────────────────────────────────────────────────────────────────────────
def fig_generalization(df):
    POLS = ["sac-topsis", "td3-topsis", "ppo-topsis", "local", "greedy", "sac-base"]
    df_f = df[df["politica"].isin(POLS)]
    agg = df_f.groupby(["tps", "politica"])["taxa_deadline"].mean().reset_index()
    pivot = agg.pivot(index="tps", columns="politica", values="taxa_deadline")

    STYLES = {
        "sac-topsis":  ("-",  "o", 2.5, 5.0),
        "td3-topsis":  ("--", "s", 1.6, 4.0),
        "ppo-topsis":  (":",  "^", 1.6, 4.0),
        "local":       ("--", "D", 1.4, 3.5),
        "greedy":      ("-.", "x", 1.4, 3.5),
        "sac-base":    (":",  "v", 1.4, 3.5),
    }

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    for pol in POLS:
        if pol not in pivot.columns:
            continue
        ls, mk, lw, ms = STYLES.get(pol, ("-", "o", 1.4, 4.0))
        color = PAL.get(pol, "#999")
        ax.plot(pivot.index, pivot[pol],
                linestyle=ls, marker=mk, color=color,
                linewidth=lw, markersize=ms,
                label=LABELS.get(pol, pol).replace("\n", " "))

    ax.set_xlabel("Vehicular density (tasks/s)")
    ax.set_ylabel("Deadline compliance rate (%)")
    ax.set_title("Zero-Shot Generalization\nAcross Vehicular Densities", fontsize=8.5)
    ax.set_xticks([10, 15, 20, 25, 30])
    ax.legend(fontsize=6.5, loc="upper right", framealpha=0.9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

    # Annotation: trained on 10 TPS
    ax.axvline(x=10, color="#666", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.text(10.3, ax.get_ylim()[0] + 1.5, "train\ndensity", fontsize=6, color="#666")

    save(fig, "fig_03_generalization.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 4 — Distribuição dos modos de execução (stacked bar)
# ─────────────────────────────────────────────────────────────────────────────
def fig_modes(df):
    POLS = ["sac-topsis", "td3-topsis", "ppo-topsis", "ppo-base",
            "local", "greedy-unfair", "greedy", "sac-base"]
    agg = df[df["politica"].isin(POLS)].groupby("politica")[
        ["pct_local", "pct_rsu", "pct_v2v"]
    ].mean().reset_index()
    agg["politica"] = pd.Categorical(agg["politica"], categories=POLS, ordered=True)
    agg = agg.sort_values("politica").reset_index(drop=True)
    labels = [LABELS.get(p, p).replace("\n", " ") for p in agg["politica"]]

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    x = np.arange(len(agg))
    w = 0.6
    b1 = ax.bar(x, agg["pct_local"], w, label="Local",  color="#607D8B")
    b2 = ax.bar(x, agg["pct_rsu"],   w, label="RSU",    color="#1976D2",
                bottom=agg["pct_local"])
    b3 = ax.bar(x, agg["pct_v2v"],   w, label="V2V",    color="#43A047",
                bottom=agg["pct_local"] + agg["pct_rsu"])

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=6.5)
    ax.set_ylabel("Proportion (%)")
    ax.set_title("Execution Mode Distribution\n(Mean across all scenarios)", fontsize=8.5)
    ax.set_ylim(0, 112)
    ax.legend(fontsize=7, loc="upper right")
    save(fig, "fig_04_modes.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 5 — Curvas de treinamento: SAC-TOPSIS vs SAC-Base vs TD3-TOPSIS
# ─────────────────────────────────────────────────────────────────────────────
def fig_training():
    agents = {
        "sac_model_topsis":    ("SAC-TOPSIS (proposed)", PAL["sac-topsis"],  "-",  2.2),
        "sac_model_base":      ("SAC-Base",              PAL["sac-base"],    "--", 1.6),
        "td3_model_topsis":    ("TD3-TOPSIS",            PAL["td3-topsis"],  "-.", 1.6),
        "ppo_model_topsis":    ("PPO-TOPSIS",            PAL["ppo-topsis"],  ":",  1.4),
    }

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.8), sharey=False)
    fig.subplots_adjust(wspace=0.35)

    for key, (label, color, ls, lw) in agents.items():
        csvpath = os.path.join(MODELS_DIR, f"{key}_train.csv")
        if not os.path.isfile(csvpath):
            continue
        d = pd.read_csv(csvpath)
        ep = d["episode"]
        reward_sm = smooth(d["mean_reward"])
        deadline_sm = smooth(d["deadline_rate"] * 100)

        axes[0].plot(ep, reward_sm, label=label, color=color,
                     linestyle=ls, linewidth=lw, alpha=0.9)
        axes[1].plot(ep, deadline_sm, label=label, color=color,
                     linestyle=ls, linewidth=lw, alpha=0.9)

    axes[0].set_xlabel("Training episode")
    axes[0].set_ylabel("Mean reward (smoothed)")
    axes[0].set_title("(a) Mean Reward Convergence", fontsize=8.5)
    axes[0].legend(fontsize=6.5, loc="lower right")

    axes[1].set_xlabel("Training episode")
    axes[1].set_ylabel("Deadline compliance (%)")
    axes[1].set_title("(b) Deadline Rate During Training", fontsize=8.5)
    axes[1].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

    save(fig, "fig_05_training.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 6 — Ablation: decomposição causal das escolhas arquiteturais
# ─────────────────────────────────────────────────────────────────────────────
def fig_ablation(df):
    """
    Ablation: mostra o ganho de cada família DRL ao adicionar o TOPSIS.
    Grupos emparelhados: Base (sem TOPSIS) → TOPSIS (com TOPSIS).
    Famílias: SAC, TD3, PPO.
    """
    POLS = ["sac-base", "sac-topsis", "td3-base", "td3-topsis", "ppo-base", "ppo-topsis"]
    LABELS_ABL = {
        "sac-base":   "SAC\nBase",
        "sac-topsis": "SAC\nTOPSIS",
        "td3-base":   "TD3\nBase",
        "td3-topsis": "TD3\nTOPSIS",
        "ppo-base":   "PPO\nBase",
        "ppo-topsis": "PPO\nTOPSIS",
    }
    COLORS_ABL = {
        "sac-base":   "#90CAF9",
        "sac-topsis": "#0D47A1",
        "td3-base":   "#80CBC4",
        "td3-topsis": "#00695C",
        "ppo-base":   "#FFCC80",
        "ppo-topsis": "#E65100",
    }

    agg = df[df["politica"].isin(POLS)].groupby("politica").agg(
        deadline=("taxa_deadline", "mean"),
        swq=("swq", "mean"),
    ).reset_index()
    agg["politica"] = pd.Categorical(agg["politica"], categories=POLS, ordered=True)
    agg = agg.sort_values("politica").reset_index(drop=True)

    labels = [LABELS_ABL.get(p, p) for p in agg["politica"]]
    colors = [COLORS_ABL.get(p, "#999") for p in agg["politica"]]

    # Posições com espaço entre famílias
    group_gaps = [0, 0, 0.4, 0, 0.4, 0]
    x = np.cumsum([1.0 + g for g in group_gaps])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.8, 2.8), sharey=False)
    fig.subplots_adjust(wspace=0.42)

    w = 0.75
    for ax, col, title, fmt in [
        (ax1, "deadline", "(a) Deadline Compliance Rate (%)", "{:.1f}%"),
        (ax2, "swq",      "(b) SWQ Index",                   "{:.4f}"),
    ]:
        bars = ax.bar(x, agg[col], w, color=colors, edgecolor="white", linewidth=0.5)

        # Destaca TOPSIS bars com borda escura
        for bar, pol in zip(bars, agg["politica"]):
            if "topsis" in str(pol):
                bar.set_edgecolor("#222222")
                bar.set_linewidth(1.4)

        for bar, val in zip(bars, agg[col]):
            ax.text(bar.get_x() + bar.get_width()/2,
                    val + 0.008 * agg[col].max(),
                    fmt.format(val), ha="center", va="bottom", fontsize=6.2)

        # Setas mostrando o delta base → TOPSIS
        pairs = [("sac-base","sac-topsis"), ("td3-base","td3-topsis"), ("ppo-base","ppo-topsis")]
        for b_pol, t_pol in pairs:
            b_row = agg[agg["politica"] == b_pol]
            t_row = agg[agg["politica"] == t_pol]
            if b_row.empty or t_row.empty:
                continue
            bi = agg[agg["politica"] == b_pol].index[0]
            ti = agg[agg["politica"] == t_pol].index[0]
            bv = float(b_row[col])
            tv = float(t_row[col])
            bx = x[bi]
            tx = x[ti]
            mid = (bx + tx) / 2
            top = max(bv, tv) + 0.07 * agg[col].max()
            delta = tv - bv
            sign = "+" if delta >= 0 else ""
            ax.annotate("", xy=(tx, tv + 0.02 * agg[col].max()),
                        xytext=(bx, bv + 0.02 * agg[col].max()),
                        arrowprops=dict(arrowstyle="->", color="#333",
                                        lw=0.9, connectionstyle="arc3,rad=-0.3"))
            ax.text(mid, top, f"{sign}{fmt.format(delta)}",
                    ha="center", va="bottom", fontsize=5.8, color="#333",
                    style="italic")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.2)
        ax.set_title(title, fontsize=8.5)
        ax.set_ylim(0, agg[col].max() * 1.35)

        # Separadores de família
        for sep in [(x[1]+x[2])/2, (x[3]+x[4])/2]:
            ax.axvline(sep, color="#ccc", linewidth=0.7, linestyle="--")
        ax.text((x[0]+x[1])/2, agg[col].max()*1.28, "SAC", ha="center",
                fontsize=7, color="#0D47A1", weight="bold")
        ax.text((x[2]+x[3])/2, agg[col].max()*1.28, "TD3", ha="center",
                fontsize=7, color="#00695C", weight="bold")
        ax.text((x[4]+x[5])/2, agg[col].max()*1.28, "PPO", ha="center",
                fontsize=7, color="#E65100", weight="bold")

    # Legenda
    base_p  = mpatches.Patch(facecolor="#ccc",    edgecolor="white",   label="Base (no TOPSIS)")
    topsis_p= mpatches.Patch(facecolor="#555",    edgecolor="#222",    label="+ TOPSIS (proposed)")
    fig.legend(handles=[base_p, topsis_p], loc="lower center", ncol=2,
               fontsize=7, bbox_to_anchor=(0.5, -0.05), frameon=True)

    save(fig, "fig_06_ablation.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 7 — Ablação interna do SAC: decomposição causal das escolhas de design
# ─────────────────────────────────────────────────────────────────────────────
def fig_sac_ablation(df):
    """
    Mostra a cadeia causal: SAC-Base → SAC-A1 → SAC-R1 → SAC-S1 → SAC-T1 → SAC-TOPSIS
    com barras horizontais ordenadas por deadline e anotação dos deltas.
    """
    POLS = ["sac-base", "sac-a1", "sac-r1", "sac-s1", "sac-t1", "sac-topsis"]
    LABELS_SAC = {
        "sac-base":   "SAC-Base\n(11D raw, 3-action, WSM reward)",
        "sac-a1":     "SAC-A1\n(11D raw, binary action, WSM reward)",
        "sac-r1":     "SAC-R1\n(11D raw, 3-action, deadline reward)",
        "sac-s1":     "SAC-S1\n(6D manual, 3-action, WSM reward)",
        "sac-t1":     "SAC-T1\n(TOPSIS 6D, 3-action, deadline reward)",
        "sac-topsis": "SAC-TOPSIS\n(TOPSIS 6D, binary action, deadline reward)",
    }
    COLORS_SAC = {
        "sac-base":   "#EF9A9A",
        "sac-a1":     "#FFCC80",
        "sac-r1":     "#FFF176",
        "sac-s1":     "#A5D6A7",
        "sac-t1":     "#81D4FA",
        "sac-topsis": "#0D47A1",
    }

    agg = df[df["politica"].isin(POLS)].groupby("politica").agg(
        deadline=("taxa_deadline", "mean"),
        swq=("swq", "mean"),
    ).reset_index()
    agg["politica"] = pd.Categorical(agg["politica"], categories=POLS, ordered=True)
    agg = agg.sort_values("politica").reset_index(drop=True)

    labels = [LABELS_SAC.get(p, p) for p in agg["politica"]]
    colors = [COLORS_SAC.get(p, "#999") for p in agg["politica"]]
    deadlines = agg["deadline"].tolist()

    fig, ax = plt.subplots(figsize=(7.16, 3.4))
    y = np.arange(len(agg))
    bars = ax.barh(y, deadlines, color=colors, edgecolor="white",
                   height=0.65, linewidth=0.5)

    # Borda destacada no SAC-TOPSIS
    bars[-1].set_edgecolor("#FFD700")
    bars[-1].set_linewidth(2.2)

    # Anotações de valor e delta
    prev = None
    for i, (bar, val) in enumerate(zip(bars, deadlines)):
        ax.text(val + 0.4, bar.get_y() + bar.get_height()/2,
                f"{val:.2f}%", va="center", fontsize=7.5)
        if prev is not None:
            delta = val - prev
            sign = "+" if delta >= 0 else ""
            ax.text(val + 0.4, bar.get_y() - 0.05,
                    f"({sign}{delta:.2f} pp vs base)",
                    va="top", fontsize=5.8, color="#555", style="italic")
        prev = val

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("Deadline Compliance Rate (%) — higher is better", fontsize=8.5)
    ax.set_title(
        "SAC-Internal Ablation: Causal Decomposition of Design Choices\n"
        "Each variant adds exactly one component relative to SAC-Base",
        fontsize=8.5, pad=6
    )
    ax.set_xlim(0, max(deadlines) * 1.22)

    # Legenda de componentes
    patches = [
        mpatches.Patch(color=COLORS_SAC["sac-base"],   label="Baseline"),
        mpatches.Patch(color=COLORS_SAC["sac-a1"],     label="+Binary action"),
        mpatches.Patch(color=COLORS_SAC["sac-r1"],     label="+Deadline reward"),
        mpatches.Patch(color=COLORS_SAC["sac-s1"],     label="+Compact state (manual)"),
        mpatches.Patch(color=COLORS_SAC["sac-t1"],     label="+TOPSIS as feature (3-action)"),
        mpatches.Patch(color=COLORS_SAC["sac-topsis"], label="SAC-TOPSIS (full)"),
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=6.5,
              framealpha=0.9, ncol=2)

    save(fig, "fig_07_sac_ablation.pdf")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Gerando figuras em: {OUT_DIR}")
    df = load_summary()
    fig_architecture()
    fig_performance(df)
    fig_generalization(df)
    fig_modes(df)
    fig_training()
    fig_ablation(df)
    fig_sac_ablation(df)
    print(f"\nConcluido — {len(os.listdir(OUT_DIR))} arquivo(s) em {OUT_DIR}")
