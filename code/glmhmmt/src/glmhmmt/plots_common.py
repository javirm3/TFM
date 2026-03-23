from __future__ import annotations

from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator

from glmhmmt.views import _STATE_HEX

try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None


def _is_polars_df(df) -> bool:
    return pl is not None and isinstance(df, pl.DataFrame)


def _subject_df(trial_df, subj: str):
    if _is_polars_df(trial_df):
        return trial_df.filter(pl.col("subject") == subj)
    return trial_df[trial_df["subject"] == subj]


def _pick_col(columns: Sequence[str], candidates: Iterable[str]) -> str:
    for cand in candidates:
        if cand in columns:
            return cand
    raise ValueError(f"None of the candidate columns exist: {list(candidates)}")


def _default_choice_meta(num_classes: int):
    if num_classes == 2:
        return {0: "royalblue", 1: "tomato"}, {0: "L", 1: "R"}
    return {0: "royalblue", 1: "gold", 2: "tomato"}, {0: "L", 1: "C", 2: "R"}


def _state_labels_and_colors(view):
    rank_order = view.state_idx_order
    labels = [view.state_name_by_idx.get(k, f"State {k}") for k in rank_order]
    colors = [_STATE_HEX[view.state_rank_by_idx.get(int(k), int(k)) % len(_STATE_HEX)] for k in rank_order]
    return rank_order, labels, colors


def plot_state_accuracy(
    views: dict,
    trial_df,
    *,
    thresh: float = 0.5,
    performance_candidates: Sequence[str] = ("correct_bool", "performance"),
    stim_candidates: Sequence[str] = ("stimd_n", "stimulus", "ILD"),
    chance_level: float | None = None,
    stim_label: str = "nonzero stimulus",
) -> tuple[plt.Figure, pd.DataFrame]:
    subjects = list(views.keys())
    if not subjects:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig, pd.DataFrame()

    first_view = views[subjects[0]]
    state_labels = [first_view.state_name_by_idx.get(k, f"State {k}") for k in first_view.state_idx_order]
    cmap = {"All": "#999999"}
    for k in first_view.state_idx_order:
        lbl = first_view.state_name_by_idx.get(k, f"State {k}")
        rank = first_view.state_rank_by_idx.get(int(k), int(k))
        cmap[lbl] = _STATE_HEX[rank % len(_STATE_HEX)]
    x_labels = ["All"] + state_labels
    if chance_level is None:
        chance_level = 100.0 / float(first_view.num_classes)

    records = []
    for subj in subjects:
        view = views[subj]
        P = np.asarray(view.smoothed_probs)
        df_sub = _subject_df(trial_df, subj)
        perf_col = _pick_col(df_sub.columns, performance_candidates)
        stim_col = _pick_col(df_sub.columns, stim_candidates)
        hits = np.asarray(df_sub[perf_col]).astype(float)
        stim = np.asarray(df_sub[stim_col]).astype(float)
        T = min(len(P), len(hits), len(stim))
        P, hits, stim = P[:T], hits[:T], stim[:T]
        valid = np.isfinite(hits) & np.isfinite(stim) & (np.abs(stim) > 0)

        if valid.sum() > 0:
            records.append(
                {
                    "subject": subj,
                    "label": "All",
                    "acc": hits[valid].mean() * 100,
                    "n": int(valid.sum()),
                }
            )

        for k in view.state_idx_order:
            lbl = view.state_name_by_idx.get(k, f"State {k}")
            mask = valid & (P[:, k] >= thresh)
            if mask.sum() > 0:
                records.append(
                    {
                        "subject": subj,
                        "label": lbl,
                        "acc": hits[mask].mean() * 100,
                        "n": int(mask.sum()),
                    }
                )

    if not records:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig, pd.DataFrame()

    df_acc = pd.DataFrame(records)
    tbl = (
        df_acc.groupby("label")[["acc", "n"]]
        .agg({"acc": "mean", "n": "sum"})
        .reindex(x_labels)
        .rename(columns={"acc": "mean_acc (%)", "n": "total_trials"})
        .round(1)
    )

    fig, ax = plt.subplots(figsize=(2 + len(x_labels) * 1.0, 4.5))
    rng = np.random.default_rng(42)
    for li, lbl in enumerate(x_labels):
        rows = df_acc[df_acc["label"] == lbl]["acc"].dropna().values
        if len(rows) == 0:
            continue
        color = cmap.get(lbl, "k")
        box = ax.boxplot(
            rows,
            positions=[li],
            widths=0.5,
            patch_artist=True,
            showfliers=False,
            zorder=1,
        )
        for patch in box["boxes"]:
            patch.set(facecolor=color, alpha=0.25, edgecolor=color, linewidth=1.2)
        for elem in ["whiskers", "caps", "medians"]:
            for artist in box[elem]:
                artist.set(color=color, linewidth=1.2)
        jitter = rng.uniform(-0.12, 0.12, size=len(rows))
        ax.scatter(
            np.full(len(rows), li) + jitter,
            rows,
            color=color,
            alpha=0.65,
            s=28,
            zorder=3,
        )
        sem = rows.std(ddof=1) / np.sqrt(len(rows)) if len(rows) > 1 else 0.0
        ax.errorbar(
            li,
            rows.mean(),
            yerr=sem,
            fmt="o",
            color=color,
            ms=7,
            capsize=4,
            lw=1.8,
            zorder=4,
        )

    ax.axhline(chance_level, color="black", linestyle="--", linewidth=0.9, alpha=0.5)
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=20, ha="right")
    ax.set_xlabel("State")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(max(0.0, chance_level - 10), 105)
    ax.set_title(f"Per-state accuracy  (K={first_view.K}, posterior ≥ {thresh}, {stim_label})")
    fig.tight_layout()
    sns.despine(fig=fig)
    return fig, tbl


def plot_session_trajectories(
    views: dict,
    trial_df,
    *,
    session_col: str = "session",
) -> plt.Figure:
    subjects = list(views.keys())
    K = next(iter(views.values())).K if views else 2
    fig, axes = plt.subplots(len(subjects), 1, figsize=(10, 3.5 * len(subjects)), squeeze=False)

    for i, subj in enumerate(subjects):
        ax = axes[i, 0]
        P = np.asarray(views[subj].smoothed_probs)
        df_sub = _subject_df(trial_df, subj)
        sess_arr = np.asarray(df_sub[session_col])
        T = min(len(P), len(sess_arr))
        P, sess_arr = P[:T], sess_arr[:T]
        sess_ids = np.unique(sess_arr)
        if len(sess_ids) == 0:
            continue

        max_len = max(int(np.sum(sess_arr == s)) for s in sess_ids)
        traj = np.full((len(sess_ids), max_len, K), np.nan)
        for si, s in enumerate(sess_ids):
            idx = np.where(sess_arr == s)[0]
            traj[si, : len(idx), :] = P[idx, :]

        mean = np.nanmean(traj, axis=0)
        n_obs = np.sum(~np.isnan(traj[:, :, 0]), axis=0)
        sem = np.nanstd(traj, axis=0, ddof=1) / np.maximum(n_obs[:, None] ** 0.5, 1)
        x = np.arange(mean.shape[0])

        for k in views[subj].state_idx_order:
            rank = views[subj].state_rank_by_idx.get(int(k), int(k))
            color = _STATE_HEX[rank % len(_STATE_HEX)]
            valid_x = ~np.isnan(mean[:, k])
            ax.plot(
                x[valid_x],
                mean[valid_x, k],
                color=color,
                lw=2,
                label=views[subj].state_name_by_idx.get(k, f"State {k}"),
            )
            ax.fill_between(
                x[valid_x],
                (mean[:, k] - sem[:, k])[valid_x],
                (mean[:, k] + sem[:, k])[valid_x],
                color=color,
                alpha=0.25,
            )

        ax.set_ylim(0, 1)
        ax.set_xlabel("Trial within session")
        ax.set_ylabel("State probability")
        ax.set_title(f"Subject {subj} — avg. state trajectory  (n={len(sess_ids)} sessions)")
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8, frameon=False)

    fig.tight_layout()
    sns.despine(fig=fig)
    return fig


def plot_state_occupancy(
    views: dict,
    trial_df,
    *,
    session_col: str = "session",
) -> plt.Figure:
    subjects = list(views.keys())
    n_rows = len(subjects) + 1
    fig, axes = plt.subplots(n_rows, 3, figsize=(14, 3.8 * n_rows), squeeze=False)
    rng = np.random.default_rng(42)

    first_view = views[subjects[0]]
    rank_order_all, labels_all, colors_all = _state_labels_and_colors(first_view)
    overall_records: list[dict] = []
    session_records: list[dict] = []
    switch_records: list[dict] = []

    def _styled_boxplot(ax, grouped_vals: list[np.ndarray], labels: list[str], colors: list[str]) -> None:
        for pos, (vals, color) in enumerate(zip(grouped_vals, colors, strict=False)):
            vals = np.asarray(vals, dtype=float)
            if vals.size == 0:
                continue
            box = ax.boxplot(
                vals,
                positions=[pos],
                widths=0.5,
                patch_artist=True,
                showfliers=False,
            )
            for patch in box["boxes"]:
                patch.set(facecolor=color, alpha=0.25, edgecolor=color, linewidth=1.2)
            for elem in ["whiskers", "caps", "medians"]:
                for artist in box[elem]:
                    artist.set(color=color, linewidth=1.2)
            jitter = rng.uniform(-0.12, 0.12, size=vals.size)
            ax.scatter(
                np.full(vals.size, pos) + jitter,
                vals,
                color=color,
                alpha=0.6,
                s=24,
                zorder=3,
            )
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylim(0, 1)

    def _plot_switch_hist(ax, changes_per_sess: list[int], title: str) -> None:
        if not changes_per_sess:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(title)
            return
        max_chg = max(changes_per_sess)
        ax.hist(
            changes_per_sess,
            bins=np.arange(-0.5, max_chg + 1.5, 1.0),
            color="#888888",
            alpha=0.75,
            edgecolor="white",
        )
        ax.set_xlim(-0.5, max_chg + 0.5)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("# state switches / session")
        ax.set_ylabel("# sessions")
        ax.set_title(title)

    for i, subj in enumerate(subjects, start=1):
        ax_occ, ax_box, ax_chg = axes[i, 0], axes[i, 1], axes[i, 2]
        P = np.asarray(views[subj].smoothed_probs)
        df_sub = _subject_df(trial_df, subj)
        sess_arr = np.asarray(df_sub[session_col])
        T = min(len(P), len(sess_arr))
        P, sess_arr = P[:T], sess_arr[:T]
        if T == 0:
            for ax in (ax_occ, ax_box, ax_chg):
                ax.set_visible(False)
            continue
        viterbi = np.argmax(P, axis=1)

        rank_order, labels, colors = _state_labels_and_colors(views[subj])
        occ = [float(np.mean(P[:, k])) for k in rank_order]
        for pos, k in enumerate(rank_order):
            overall_records.append(
                {
                    "subject": subj,
                    "state_idx": int(k),
                    "state_label": labels[pos],
                    "occupancy": occ[pos],
                }
            )
        ax_occ.bar(labels, occ, color=colors, alpha=0.85)
        ax_occ.set_ylim(0, 1)
        ax_occ.set_ylabel("Fractional occupancy")
        ax_occ.set_title(f"Subject {subj} - overall occupancy")

        sess_occ = {int(k): [] for k in rank_order}
        changes_per_sess = []
        for s in np.unique(sess_arr):
            p_s = P[sess_arr == s]
            v = viterbi[sess_arr == s]
            if len(v) == 0:
                continue
            n_changes = int(np.sum(np.diff(v) != 0))
            changes_per_sess.append(n_changes)
            switch_records.append({"subject": subj, "session": s, "switches": n_changes})
            for pos, k in enumerate(rank_order):
                occ_s = float(np.mean(p_s[:, k]))
                sess_occ[int(k)].append(occ_s)
                session_records.append(
                    {
                        "subject": subj,
                        "session": s,
                        "state_idx": int(k),
                        "state_label": labels[pos],
                        "occupancy": occ_s,
                    }
                )

        _styled_boxplot(
            ax_box,
            [np.asarray(sess_occ[int(k)], dtype=float) for k in rank_order],
            labels,
            colors,
        )
        ax_box.set_ylabel("Session occupancy")
        ax_box.set_title(f"Subject {subj} - occupancy by session")

        _plot_switch_hist(ax_chg, changes_per_sess, f"Subject {subj} - state switches")

    ax_all_occ, ax_all_sess, ax_all_chg = axes[0, 0], axes[0, 1], axes[0, 2]
    _styled_boxplot(
        ax_all_occ,
        [
            np.asarray(
                [row["occupancy"] for row in overall_records if row["state_label"] == lbl],
                dtype=float,
            )
            for lbl in labels_all
        ],
        labels_all,
        colors_all,
    )
    ax_all_occ.set_ylabel("Fractional occupancy")
    ax_all_occ.set_title("All selected subjects - overall occupancy")

    _styled_boxplot(
        ax_all_sess,
        [
            np.asarray(
                [row["occupancy"] for row in session_records if row["state_label"] == lbl],
                dtype=float,
            )
            for lbl in labels_all
        ],
        labels_all,
        colors_all,
    )
    ax_all_sess.set_ylabel("Session occupancy")
    ax_all_sess.set_title("All selected sessions - occupancy by session")

    _plot_switch_hist(
        ax_all_chg,
        [int(row["switches"]) for row in switch_records],
        "All selected sessions - state switches",
    )

    fig.tight_layout()
    sns.despine(fig=fig)
    return fig


def plot_session_deepdive(
    views: dict,
    trial_df,
    subj: str,
    sess,
    *,
    session_col: str = "session",
    performance_candidates: Sequence[str] = ("correct_bool", "performance"),
    stim_candidates: Sequence[str] = ("stimd_n", "ILD", "stimulus"),
    response_candidates: Sequence[str] = ("response", "Choice"),
    trace_x_candidates: Sequence[str] = ("A_R", "A_L", "A_C"),
    trace_u_candidates: Sequence[str] = ("A_plus", "A_minus"),
    choice_colors: dict[int, str] | None = None,
    choice_labels: dict[int, str] | None = None,
) -> plt.Figure:
    try:
        sess = int(sess)
    except (TypeError, ValueError):
        pass

    df_sub_all = _subject_df(trial_df, subj)
    if _is_polars_df(df_sub_all):
        sess_row_indices = df_sub_all.with_row_index("_r").filter(pl.col(session_col) == sess)["_r"].to_numpy()
        df_sess = df_sub_all.filter(pl.col(session_col) == sess)
    else:
        df_sub_all = df_sub_all.reset_index(drop=True)
        sess_row_indices = df_sub_all.index[df_sub_all[session_col] == sess].to_numpy()
        df_sess = df_sub_all[df_sub_all[session_col] == sess]

    perf_col = _pick_col(df_sess.columns, performance_candidates)
    stim_col = _pick_col(df_sess.columns, stim_candidates)
    response_col = _pick_col(df_sess.columns, response_candidates)
    hits = np.asarray(df_sess[perf_col]).astype(float)
    stim = np.asarray(df_sess[stim_col]).astype(float)
    response = np.asarray(df_sess[response_col]).astype(int)

    probs_all = np.asarray(views[subj].smoothed_probs)
    probs = probs_all[sess_row_indices]
    T = min(probs.shape[0], len(hits))
    probs, hits, stim, response = probs[:T], hits[:T], stim[:T], response[:T]
    x = np.arange(T)

    X_sess = np.asarray(views[subj].X)[sess_row_indices][:T]
    X_idx = {f: i for i, f in enumerate(views[subj].feat_names)}
    trace_sources = {}
    if views[subj].U is not None:
        U_sess = np.asarray(views[subj].U)[sess_row_indices][:T]
        U_idx = {f: i for i, f in enumerate(views[subj].U_cols)}
        for name in trace_u_candidates:
            if name in U_idx:
                trace_sources[name] = (U_sess, U_idx[name])
    for name in trace_x_candidates:
        if name in X_idx and name not in trace_sources:
            trace_sources[name] = (X_sess, X_idx[name])

    rolling_acc = np.full(T, np.nan)
    window = 20
    nz = np.abs(stim) > 0
    for ti in range(T):
        start = max(0, ti - window + 1)
        window_mask = nz[start : ti + 1]
        if np.any(window_mask):
            rolling_acc[ti] = 100.0 * hits[start : ti + 1][window_mask].mean()

    if choice_colors is None or choice_labels is None:
        choice_colors, choice_labels = _default_choice_meta(views[subj].num_classes)

    rank_order, _labels, colors = _state_labels_and_colors(views[subj])
    engaged_k = views[subj].engaged_k()

    n_rows = 2 if trace_sources else 1
    height_ratios = [2, 1.5] if trace_sources else [1]
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(14, 5 + 2.5 * (n_rows - 1)),
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
    )
    axes = np.atleast_1d(axes)
    ax1 = axes[0]

    bottom = np.zeros(T)
    for pos, k in enumerate(rank_order):
        color = colors[pos]
        ax1.fill_between(
            x,
            bottom,
            bottom + probs[:, k],
            alpha=0.7,
            color=color,
            label=views[subj].state_name_by_idx.get(k, f"State {k}"),
        )
        bottom += probs[:, k]

    ax1.plot(
        x,
        probs[:, engaged_k],
        color=_STATE_HEX[0],
        lw=2,
        label=f"P({views[subj].state_name_by_idx.get(engaged_k, 'Engaged')})",
    )

    for resp, color in choice_colors.items():
        mask = response == resp
        if mask.sum() == 0:
            continue
        ax1.scatter(
            x[mask],
            np.ones(mask.sum()) * 1.03,
            c=color,
            s=5,
            marker="|",
            label=choice_labels.get(resp, str(resp)),
            transform=ax1.get_xaxis_transform(),
            clip_on=False,
        )

    ax1.set_ylim(0, 1)
    ax1.set_ylabel("State probability")
    ax1.set_title(f"Subject {subj}  —  session {sess}  ({T} trials)")

    ax1r = ax1.twinx()
    ax1r.plot(x, rolling_acc, color="black", lw=1.8, linestyle="-", alpha=0.7, label="Rolling accuracy (5 trials)")
    ax1r.axhline(100.0 / float(views[subj].num_classes), color="grey", lw=0.9, linestyle="--", alpha=0.5)
    ax1r.set_ylim(0, 105)
    ax1r.set_ylabel("Accuracy (%)", color="black")

    lines1, labs1 = ax1.get_legend_handles_labels()
    lines1r, labs1r = ax1r.get_legend_handles_labels()
    ax1.legend(lines1 + lines1r, labs1 + labs1r, bbox_to_anchor=(1.08, 1), loc="upper left", fontsize=8, frameon=False)

    trace_colors = {
        "A_plus": "royalblue",
        "A_minus": "gold",
        "A_L": "royalblue",
        "A_C": "gold",
        "A_R": "tomato",
    }
    if trace_sources:
        ax2 = axes[1]
        for name, (arr, ci) in trace_sources.items():
            ax2.plot(x, arr[:, ci], label=name, color=trace_colors.get(name, "gray"), lw=1.5, alpha=0.85)
        ax2.set_ylabel("Action trace")
        ax2.set_ylim(0, None)
        ax2.set_xlabel("Trial within session")
        ax2.legend(bbox_to_anchor=(1.08, 1), loc="upper left", fontsize=8, frameon=False)
    else:
        ax1.set_xlabel("Trial within session")

    fig.tight_layout()
    fig.subplots_adjust(right=0.82)
    sns.despine(fig=fig, right=False)
    return fig
