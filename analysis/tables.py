#!/usr/bin/env python3
"""LaTeX tables of the revised paper, generated from ``results/summaries``.

table_estimators.tex   Table 4   cross-channel estimators and the calibration gate
table_recovery.tex     Table 5   dropout loss and recovered fraction per dataset
table_breadth.tex      Table 6   held-out configurations and TiRex
table_s_holm.tex       Supplementary: Holm-adjusted Wilcoxon tests
table_s_gate.tex       Supplementary: calibration gate per channel
table_s_channels.tex   Supplementary: channel-level recovery (TimesFM 2.5)
"""

from __future__ import annotations

import math

import numpy as np
from common import NAMES, RESULTS, read_json

OUT = RESULTS / "tables"
LOSSES = ("mse", "mae", "quantile_loss")


def num(v: float) -> str:
    return f"${v:+.2f}$" if v > 0 else f"${v:.2f}$"


def cell(summary: dict, loss: str) -> str:
    m = summary["metrics"][loss]
    lo, hi = m["dataset_cluster_bootstrap_ci95_percent"]
    return f"{num(m['geometric_relative_change_percent'])} [{num(lo)}, {num(hi)}]"


def row(label: str, gate: str, summary: dict) -> str:
    m = summary["metrics"]["mse"]
    return (
        f"{label} & {gate} & "
        + " & ".join(cell(summary, loss) for loss in LOSSES)
        + f" & {m['wins']}/{m['losses']} \\\\"
    )


def effect_table(caption: str, label: str, first: str, second: str, body: list[str]) -> str:
    return (
        "\\begin{table}[t]\n\\centering\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\small\n\\resizebox{{\\textwidth}}{{!}}{{%\n"
        "\\begin{tabular}{llrrrr}\n\\toprule\n"
        f"{first} & {second} & MSE $\\Delta$ (\\%) [95\\% CI] & MAE $\\Delta$ (\\%) [95\\% CI] & "
        "QL $\\Delta$ (\\%) [95\\% CI] & MSE $W/L$ \\\\\n\\midrule\n"
        + "\n".join(body)
        + "\n\\bottomrule\n\\end{tabular}\n}\n\\end{table}\n"
    )


def estimators_table() -> str:
    c = read_json("estimators.json")["primary"]
    names = (
        ("Ridge", "ridge_ungated", "ridge_gated"),
        ("Ledoit--Wolf", "lw_ungated", "lw_gated"),
        ("ContextGuard", "contextguard", "contextguard_adaptive"),
    )
    body = ["\\multicolumn{6}{l}{\\emph{Relative to topology}} \\\\"]
    for name, u, g in names:
        body += [row(name, "none", c[f"{u}__vs__topology"]), row(name, "calibration", c[f"{g}__vs__topology"])]
    body += ["\\midrule", "\\multicolumn{6}{l}{\\emph{Gated relative to ungated}} \\\\"]
    body += [row(name, "", c[f"{g}__vs__{u}"]) for name, u, g in names]
    body += [
        "\\midrule",
        "\\multicolumn{6}{l}{\\emph{Gated ContextGuard relative to gated estimator}} \\\\",
        row("Ridge", "", c["contextguard_adaptive__vs__ridge_gated"]),
        row("Ledoit--Wolf", "", c["contextguard_adaptive__vs__lw_gated"]),
    ]
    return effect_table(
        "Cross-channel estimators relative to topology-aware repair and the effect of the "
        "calibration gate (160 primary cells).",
        "tab:estimators",
        "Estimator",
        "Gate",
        body,
    )


def breadth_table() -> str:
    b = read_json("breadth.json")
    body = [
        row("Held-out (hourly, 6 configs)", "all three", b["heldout"]["all"]),
        row("", "Chronos-2", b["heldout"]["chronos2"]),
        row("", "TimesFM 2.5", b["heldout"]["timesfm25"]),
        row("", "TiRex", b["heldout"]["tirex"]),
        row("Primary (8 configs)", "TiRex", b["tirex_primary"]),
        "\\midrule",
        "\\multicolumn{6}{l}{\\emph{Gated relative to ungated ContextGuard}} \\\\",
        row("Held-out (hourly, 6 configs)", "all three", b["gate"]["heldout"]),
        row("Primary (8 configs)", "TiRex", b["gate"]["tirex_primary"]),
    ]
    return effect_table(
        "ContextGuard relative to topology on six held-out configurations and with a third forecaster.",
        "tab:breadth",
        "Set",
        "Forecaster(s)",
        body,
    )


RECOVERY_ORDER = (
    "bitbrains_fast_5T",
    "bitbrains_rnd_5T",
    "bizitobs_application",
    "bizitobs_l2c_5T",
    "jena_10T",
    "ett1_15T",
    "ett2_15T",
    "bizitobs_service",
)


def recovery_table() -> str:
    units = read_json("determinants.json")["primary_dataset_level"]["units"]
    get = {(u["dataset"], u["model"]): u for u in units if u["metric"] == "mse"}
    channels = {
        ds: v["channels"]
        for ds, v in read_json("gate_validation.json")["primary"]["crossfit_agreement"]["datasets"].items()
    }
    lines = []
    for ds in RECOVERY_ORDER:
        c, t = get[(ds, "chronos2")], get[(ds, "timesfm25")]
        lines.append(
            f"{NAMES[ds]} & {channels[ds]} & {100 * c['loss_increase']:.1f} & {100 * t['loss_increase']:.1f} & "
            f"{100 * c['recovered']:.1f} & {100 * t['recovered']:.1f} & {c['mean_I']:.3f} \\\\"
        )
    return (
        "\\begin{table}[t]\n\\centering\n\\caption{Dropout-induced MSE loss and the fraction recovered by "
        "ContextGuard, with training-only calibration reliability.}\n\\label{tab:recovery}\n\\small\n"
        "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{lrrrrrr}\n\\toprule\n"
        " & & \\multicolumn{2}{c}{MSE increase (\\%)} & "
        "\\multicolumn{2}{c}{Recovered (\\%)} & \\\\\n"
        "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\n"
        "Dataset & $C$ & Chronos-2 & TimesFM 2.5 & Chronos-2 & TimesFM 2.5 & Mean $I_j$ \\\\\n\\midrule\n"
        + "\n".join(lines)
        + "\n\\bottomrule\n\\end{tabular}\n}\n\\end{table}\n"
    )


def pvalue(p: float) -> str:
    if p == 0:
        return "$<10^{-300}$"
    e = int(math.floor(math.log10(p)))
    return f"${p / 10**e:.2f}\\times10^{{{e}}}$" if p < 1e-3 else f"{p:.3f}"


def holm_table() -> str:
    c = read_json("estimators.json")["primary"]
    helix = read_json("original/helix.json")["comparisons"]
    entries = [
        ("ContextGuard vs topology", c["contextguard_adaptive__vs__topology"]),
        ("HELIX vs topology", helix["helix_vs_topology"]["overall"]),
        ("ContextGuard vs HELIX", helix["contextguard_vs_helix"]["overall"]),
    ]
    labels = (
        ("contextguard", "topology", "ContextGuard (ungated) vs topology"),
        ("ridge_ungated", "topology", "Ridge (ungated) vs topology"),
        ("ridge_gated", "topology", "Ridge (gated) vs topology"),
        ("lw_ungated", "topology", "Ledoit--Wolf (ungated) vs topology"),
        ("lw_gated", "topology", "Ledoit--Wolf (gated) vs topology"),
        ("contextguard_adaptive", "contextguard", "ContextGuard gated vs ungated"),
        ("ridge_gated", "ridge_ungated", "Ridge gated vs ungated"),
        ("lw_gated", "lw_ungated", "Ledoit--Wolf gated vs ungated"),
        ("contextguard_adaptive", "ridge_gated", "ContextGuard vs gated ridge"),
        ("contextguard_adaptive", "lw_gated", "ContextGuard vs gated Ledoit--Wolf"),
        ("contextguard_crossfit", "contextguard_adaptive", "Cross-fitted vs published gate"),
    )
    entries += [(lab, c[f"{m}__vs__{b}"]) for m, b, lab in labels]
    body = "\n".join(
        f"{lab} & " + " & ".join(pvalue(s["metrics"][loss]["wilcoxon_less_holm_p"]) for loss in LOSSES) + " \\\\"
        for lab, s in entries
    )
    return (
        "\\begin{table}[H]\n\\centering\n\\caption{One-sided paired Wilcoxon tests, Holm-adjusted across the "
        "three losses (160 cells).}\n\\label{tab:s-holm}\n\\small\n\\begin{tabular}{lrrr}\n\\toprule\n"
        "Comparison & MSE & MAE & Quantile loss \\\\\n\\midrule\n"
        + body
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


def longtable(columns: str, caption: str, label: str, header: str, lines: list[str]) -> str:
    """Multi-page table: caption and label on the first page, the header repeated on every page."""
    head = f"\\toprule\n{header} \\\\\n\\midrule\n"
    return (
        f"\\begin{{longtable}}{{{columns}}}\n\\caption{{{caption}}}\\label{{{label}}}\\\\\n{head}\\endfirsthead\n"
        f"{head}\\endhead\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{longtable}\n"
    )


def by_name(datasets):
    return sorted(datasets, key=lambda d: NAMES[d])


def gate_table() -> str:
    primary = read_json("gate_validation.json")["primary"]
    ratios = primary["test_context_reconstruction"]["ratios"]
    gates = primary["crossfit_agreement"]["datasets"]
    lines = []
    for ds in by_name(gates):
        g = gates[ds]
        for c in range(g["channels"]):
            r = ratios.get(f"{ds}|{c}")
            lines.append(
                f"{NAMES[ds]} & {c} & {g['I'][c]:.3f} & {g['g'][c]:.2f} & {g['I_crossfit'][c]:.3f} & "
                f"{g['g_crossfit'][c]:.2f} & " + (f"{r:.3f}" if r is not None else "--") + " \\\\"
            )
    header = "Dataset & Ch. & $I_j$ & $g_j$ & $I_j^{\\mathrm{cf}}$ & $g_j^{\\mathrm{cf}}$ & Test ratio"
    return longtable("lrrrrrr", "Calibration gate per channel.", "tab:s-gate", header, lines)


def channels_table() -> str:
    units = read_json("determinants.json")["primary_channel_level_timesfm25"]["units"]
    lines = []
    for u in sorted(units, key=lambda u: (NAMES[u["dataset"]], u["channel"])):
        rec = "--" if not np.isfinite(u["recovered"]) else f"{100 * u['recovered']:.1f}"
        lines.append(
            f"{NAMES[u['dataset']]} & {u['channel']} & {u['windows']} & {100 * u['loss_increase']:.1f} & {rec} & "
            f"{u['R2']:.3f} & {u['I']:.3f} & {u['mean_abs_r']:.3f} & {u['max_abs_r']:.3f} \\\\"
        )
    header = "Dataset & Ch. & $n$ & Loss (\\%) & Rec. (\\%) & $R^2$ & $I_j$ & mean$|r|$ & max$|r|$"
    caption = "Channel-level recovery with TimesFM 2.5 on the primary configurations."
    return longtable("lrrrrrrrr", caption, "tab:s-channels", header, lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, build in (
        ("table_estimators", estimators_table),
        ("table_recovery", recovery_table),
        ("table_breadth", breadth_table),
        ("table_s_holm", holm_table),
        ("table_s_gate", gate_table),
        ("table_s_channels", channels_table),
    ):
        (OUT / f"{name}.tex").write_text(build())
        print(f"results/tables/{name}.tex")


if __name__ == "__main__":
    main()
