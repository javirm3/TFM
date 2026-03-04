import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return


@app.cell
def _():
    import polars as pl
    import pandas as pd
    import numpy as np
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    import paths
    from scripts.alexis_functions import filter_behavior

    return filter_behavior, np, paths, pd, pl


@app.cell
def _(filter_behavior, paths, pl):
    base_path = paths.DATA_PATH / "Alexis"
    experiment_folders = [f"2AFC_{i}" for i in range(1, 7)] + ["2AFC"]

    dfs = []
    for folder in experiment_folders:
        folder_path = base_path / folder
        if folder_path.exists():
            for csv_file in folder_path.glob("*.csv"):
                if "corrupted" not in csv_file.name:
                    df = (
                        pl.read_csv(csv_file, infer_schema=False)
                    )
                    df = df.drop("Experiment")
                    df = df.with_columns(pl.lit(folder).alias("Experiment"))
                    dfs.append(df)

    combined_df = pl.concat(dfs, how="diagonal")

    # Cast columns back to their proper types
    combined_df = combined_df.with_columns(
        pl.col(col).cast(pl.Float64, strict=False)
        for col in combined_df.columns
        if col not in ("experiment",)  # keep string cols as-is
        and combined_df[col].cast(pl.Float64, strict=False).null_count() == combined_df[col].null_count()
    )
    combined_df = combined_df.filter((pl.col("Experiment").is_in(['2AFC_2', '2AFC_3', '2AFC_4'])))
    combined_df = combined_df.rename({"Subject" : "subject"})
    combined_df = combined_df.with_columns(pl.col("subject").cast(pl.Utf8))
    combined_df = combined_df.select(
        [c for c in ["subject", "Trial", "Side", "Choice", "Hit", "Punish", "Session", "ILD", "Filename", "Experiment", "Task", "P" ,"p", "Condition", "AW", "WarmUp", "Date"]
         if c in combined_df.columns]
    )
    output_path = paths.DATA_PATH / "alexis_combined.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df = pl.concat([
        pl.from_pandas(
            filter_behavior(
                combined_df.filter(pl.col("subject") == subj).to_pandas()
            )
        )
        for subj in combined_df["subject"].unique().to_list()
    ])
    combined_df.write_parquet(output_path)
    print(f"Saved to {output_path}")
    print(combined_df.group_by("subject").agg([pl.len().alias("n_trials")]).sort("n_trials"))
    return


app._unparsable_cell(
    r"""
    Apply filter_behavior per subject per experiment; for 2AFC_6 tag each row with condition.
        parts = []
        for (exp,), df_group in combined_df.group_by(["Experiment"], maintain_order=True):
       if exp not in ['2AFC_2', '2AFC_3', '2AFC_4', '2AFC_6']:
           continue
       subj_parts = []
       for subj, df_subj in df_group.to_pandas().groupby('Subject', sort=False):
           df_subj = df_subj.reset_index(drop=True)
           df_subj = filter_behavior(df_subj, clean_start=True, drop_miss=True, filter_drug=False)
           subj_parts.append(df_subj)
       df_pd = pd.concat(subj_parts, ignore_index=True)
       if exp == '2AFC_6':
           # Default: rest (no drug, no saline)
           df_pd['condition'] = 'rest'
           # Identify paired sessions per subject (saline immediately followed by drug)
           if 'Date' in df_pd.columns:
               for subj, df_subj in df_pd.groupby('Subject'):
                   df_paired = filter_drug_sessions(df_subj.copy())
                   paired_dates = set(df_paired['Date'].unique())
                   mask = (df_pd['Subject'] == subj) & df_pd['Date'].isin(paired_dates)
                   df_pd.loc[mask & (df_pd['Drug'] == 0), 'condition'] = 'saline'
                   df_pd.loc[mask & (df_pd['Drug'] == 1), 'condition'] = 'drug'
       else:
           df_pd['condition'] = 'rest'
       parts.append(pl.from_pandas(df_pd))
        combined_df_filtered = pl.concat(parts, how="diagonal")
        combined_df_filtered
    """,
    name="_"
)


@app.cell
def _(combined_df_filtered, pl):
    # Build an explicit rename dict: every column to lowercase, except ILD stays uppercase
    _rename_map = {
        col: col if col == "ILD" else col.lower()
        for col in combined_df_filtered.columns
    }
    combined_df_reduced = combined_df_filtered.rename(_rename_map)

    # If session is still missing (e.g. all-null or name mismatch), derive from filename per subject
    if "session" not in combined_df_reduced.columns:
        print("Sessopm is missing")
        combined_df_reduced = combined_df_reduced.with_columns(
            pl.col("filename")
              .rank("dense")
              .over(["subject", "experiment"])
              .cast(pl.Int32)
              .alias("session")
        )

    combined_df_reduced = combined_df_reduced.select(
        [c for c in ["subject", "trial", "side", "choice", "hit", "punish", "session", "ILD", "filename", "experiment", "p", "condition"]
         if c in combined_df_reduced.columns]
    )
    combined_df_reduced
    return (combined_df_reduced,)


@app.cell
def _(pd):
    def clean_session_start(df):
        """
        Remove AW and WarmUp trials from one or multiple sessions.
        :param df: DataFrame containing one or more sessions
        :return: Cleaned DataFrame
        """
        def _clean(group):
            if 'AW' not in group.columns or 'WarmUp' not in group.columns:
                return group
            aw_vals = group['AW'].dropna()
            warmup_vals = group['WarmUp'].dropna()
            if aw_vals.empty or warmup_vals.empty:
                return group
            aw = int(aw_vals.unique()[0])
            warmup = int(warmup_vals.unique()[0])

            warmup_len = 40
            if warmup == 1:
                cleaned = group.iloc[warmup_len:]
            else:
                cleaned = group.copy()

            if warmup == 0 and aw > 0:
                cleaned = cleaned.iloc[aw:]

            return cleaned

        # Remove the AW and Warm Up trials
        _ = len(df)
        cleaned_groups = [_clean(group) for _, group in df.groupby('Session', sort=False)]
        df = pd.concat(cleaned_groups, ignore_index=True) if cleaned_groups else df
        print(f'Removed {(_ - len(df))} trials from session start (AW and Warm Up trials)')

        return df
    def filter_drug_sessions(df):
        """Filter out saline sessions (Drug==0) that are not immediately followed by a drug session (Drug==1) for paired
        saline-drug analyses (batch #6).
        :return: df with only paired saline-drug sessions
        """

        # The column date is called differently depending on session or intersession data
        if 'Date' in df.columns:
            col_name = 'Date'  # Sessions
            # df.drop_duplicates(subset=col_name, inplace=True)  # Keep a row per unique date
        elif 'Dates' in df.columns:
            col_name = 'Dates'  # Intersessions

        # In case of sessions data
        df[col_name] = pd.to_datetime(df[col_name])  # Convert to datetime if not already
        df.sort_values(by=col_name, inplace=True)  # Sort by date
        df.reset_index(drop=True, inplace=True)  # Reset index inplace

        # Find saline sessions immediately followed by a drug session
        paired_sessions = []
        for i in range(len(df) - 1):
            current = df.iloc[i]
            next = df.iloc[i + 1]
            if current.Drug == 0 and next.Drug == 1:
                paired_sessions.append(df[col_name][i])

        # Filter original df
        df = df[(df[col_name].isin(paired_sessions) | (df.Drug == 1))]
        df.reset_index(drop=True, inplace=True)

        # Get summary df of paired sessions
        # summary = (df[['Date', 'Drug']].drop_duplicates(['Date', 'Drug']).sort_values('Date').reset_index(drop=True))

        return df


    def filter_behavior(df, clean_start=True, drop_miss=True, filter_drug=True):
        """
        Filter the behavior DataFrame for one subject.
        :param df: DataFrame containing the data
        :return: Filtered DataFrame
        """

        _ = len(df)

        # General filters
        # Remove AW and WarmUp trials
        if clean_start:
            df = clean_session_start(df)
        # Drop misses (Choice == NaN)
        if drop_miss:
            df = df.dropna(subset=['Choice']).reset_index(drop=True)

        # Experiment-specific filters
        experiment = df.Experiment.unique()[0]

        if experiment in ['2AFC_2', '2AFC_3']:
            # These 3 conditons return 0 trials
            # df = df[df.Stage == 4].reset_index(drop=True)
            # df = df[df.Motor == 4].reset_index(drop=True)
            # df = df[df.StimDur == 1].reset_index(drop=True)
            df = df[df.P > 0].reset_index(drop=True)

        elif experiment in ['2AFC_4', '2AFC_6']:
            # These 3 conditons return 0 trials
            # df = df[df.Task == 'FD'].reset_index(drop=True)  # (otherwise bump in lick rate before stim. onset)
            # df = df[df.StimDur == 1].reset_index(drop=True)
            # df = df[df.Delay == 0.5].reset_index(drop=True)
            df = df[df.P > 0].reset_index(drop=True)

            if experiment == '2AFC_6' and filter_drug:  # Drug group
                df = filter_drug_sessions(df)

        elif experiment == '2AFC_5':  # Ephys group
            df = df[df.Task == 'FD'].reset_index(drop=True)
            df = df[df.StimDur == 0.5].reset_index(drop=True)
            df = df[df.Delay == 0.5].reset_index(drop=True)
            df = df[df.P == 0].reset_index(drop=True)

        print(f'Total:{round((_ - len(df)) / 1000)}k trials')

        return df

    return (filter_behavior,)


@app.cell
def _(np, paths, pd):
    def get_ild(stim_set=2):
        if stim_set == 1:
            sounds_path = paths.DATA_PATH / "Alexis" / "sounds_1.csv"
        elif stim_set == 2:
            sounds_path = paths.DATA_PATH / "Alexis" / "sounds_2.csv"
        elif stim_set == 6:
            sounds_path = paths.DATA_PATH / "Alexis" / "sounds_6.1.csv"
        sounds = pd.read_csv(sounds_path)
        n_frames = sounds.n_frames.unique()[0]
        frames_ild = (sounds[[f"ER{n}" for n in range(n_frames)]].values
                      - sounds[[f"EL{n}" for n in range(n_frames)]].values)
        frames_ild = pd.DataFrame(frames_ild)
        frames_ild.insert(0, "filename", sounds.filename)
        return frames_ild, sounds, n_frames

    def make_frames_dm(df_pd, experiment, residuals=True, zscore=False):
        stim_set = 6 if experiment == "2AFC_6" else 2
        frames_ild, sounds, n_frames = get_ild(stim_set=stim_set)
        if residuals:
            sounds_ild = sounds.ILD
            first_frame = frames_ild[0].copy()
            first_frame.iloc[0] = 0
            first_frame.iloc[-1] = 0
            if stim_set == 6:
                frames_ild = frames_ild.drop(["filename", 0], axis=1).sub(sounds_ild, axis="rows")
                frames_ild.insert(0, "filename", sounds.filename)
                frames_ild.insert(1, 0, first_frame)
            else:
                frames_ild = frames_ild.drop("filename", axis=1).sub(sounds_ild, axis="rows")
                frames_ild.insert(0, "filename", sounds.filename)
        filenames = df_pd["filename"].tolist()
        idx = [np.where(sounds.filename.values == f)[0][0] for f in filenames]
        stim_strength = frames_ild.iloc[idx].drop(columns=["filename"])
        stim_strength = stim_strength.reset_index(drop=True)
        if (not residuals) and zscore:
            from scipy import stats
            stim_strength = pd.DataFrame(stats.zscore(stim_strength, axis=0))
        return stim_strength, n_frames

    def make_net_ild_dm(df_pl):
        """Build one ±1/0 column per unique nonzero |ILD| using Polars."""
        import polars as pl
        unique_ilds = sorted(
            int(v) for v in df_pl["ILD"].abs().unique().to_list() if v is not None and v != 0
        )
        return df_pl.with_columns([
            pl.when(pl.col("ILD") == ild).then(pl.lit(1))
              .when(pl.col("ILD") == -ild).then(pl.lit(-1))
              .otherwise(pl.lit(0))
              .cast(pl.Int8)
              .alias(f"net_ild_{ild}")
            for ild in unique_ilds
        ])

    return make_frames_dm, make_net_ild_dm


@app.cell
def _(combined_df_reduced, make_frames_dm, make_net_ild_dm, pl):
    # ── stim_vals: ILD normalised to [-1, 1] per session ──────────────
    df_features = combined_df_reduced.with_columns(
        (pl.col("ILD") / pl.col("ILD").abs().max().over("session"))
        .cast(pl.Float32).alias("stim_vals")
    )

    # ── net_ild columns ────────────────────────────────────────────────────
    df_features = make_net_ild_dm(df_features)

    # ── stim_strength (frames DM) via pandas bridge ────────────────────────
    frames_parts = []
    for (experiment, session), _df_group in df_features.group_by(
        ["experiment", "session"], maintain_order=True
    ):
        _df_pd = _df_group.to_pandas()
        stim_strength, n_frames = make_frames_dm(_df_pd, experiment=experiment)
        stim_strength.columns = [f"sf_{c}" for c in stim_strength.columns]
        max_val = stim_strength.values.max()
        if max_val != 0:
            stim_strength = stim_strength / max_val
        frames_parts.append(
            pl.concat([_df_group, pl.from_pandas(stim_strength.reset_index(drop=True))], how="horizontal")
        )
    df_features = pl.concat(frames_parts, how="diagonal")

    df_features
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
