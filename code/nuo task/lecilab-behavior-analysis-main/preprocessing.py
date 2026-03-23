import marimo

__generated_with = "0.21.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return


@app.cell
def _():
    import lecilab_behavior_analysis.utils as utils
    import lecilab_behavior_analysis.df_transforms as dft
    import lecilab_behavior_analysis.plots as plots
    from pathlib import Path
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from IPython.display import clear_output
    import time
    import datetime

    return Path, clear_output, dft, np, pd, plots, time, utils


@app.cell
def _(utils):
    tv_projects = utils.get_server_projects()
    for i, p in enumerate(tv_projects):
        print(f"[{i}] {p}")
    return (tv_projects,)


@app.cell
def _(tv_projects):
    # select a project
    project = tv_projects[2]
    print(f"Selected project: {project}")
    return (project,)


@app.cell
def _(project, utils):
    # see the available animals
    animals = utils.get_animals_in_project(project)
    # optionally, remove some animals
    animals_to_remove = ['test', 'test2', 'punish_test', 'prueba']
    animals = [animal for animal in animals if animal not in animals_to_remove]
    print(animals)
    return (animals,)


@app.cell
def _(Path, animals, clear_output, project, time, utils):
    # retrieve the data for the remaining animals
    for mouse in animals:
        local_path = Path(utils.get_outpath()) / Path(project) / Path("sessions") / Path(mouse)
        # create the directory if it doesn't exist
        local_path.mkdir(parents=True, exist_ok=True)
        # download the session data
        utils.rsync_cluster_data(
            project_name=project,
            file_path="sessions/{}/{}.csv".format(mouse, mouse),
            local_path=str(local_path),
            credentials=utils.get_idibaps_cluster_credentials(),
        )
    # Clear the output of the jupyter cell
    clear_output(wait=True)
    time.sleep(.5)
    print("Data downloaded successfully.")
    return


@app.cell
def _(Path, animals, clear_output, dft, pd, project, time, utils):
    # project = "visual_and_COT_data"
    # animals=['ACV001', 'ACV002', 'ACV003', 'ACV004', 'ACV005', 'ACV006', 'ACV007', 'ACV008', 'ACV009', 'ACV010']
    df_list = []
    for _mouse in animals:
        _local_path = Path(utils.get_outpath()) / Path(project) / Path("sessions") / Path(_mouse)
        df = pd.read_csv(_local_path / Path(f'{_mouse}.csv'), sep=";")
        df_list.append(df)
        print(f"Loaded data for {_mouse}.")
    # concatenate the dataframes
    df = pd.concat(df_list, ignore_index=True)
    clear_output(wait=True)
    time.sleep(.5)
    print("Data read successfully.")
    df = dft.analyze_df(df)
    print("Dataframe analyzed.")

    return (df,)


@app.cell
def _(df, dft, plots, utils):
    # tests on the auditory behavior
    aud_df = df[df.current_training_stage == 'TwoAFC_auditory_hard']
    aud_df = dft.get_performance_by_difficulty_ratio(aud_df)

    aud_df['col_test'] = aud_df['percentage_of_timebins_with_evidence_high'].apply(lambda x:(10 * (x - (1 - x))))
    # apply a log respecting the sign of the value
    # aud_df['col_test'] = aud_df['col_test'].apply(lambda x: np.log(x) if x > 0 else -np.log(-x))
    # plot the psychometric
    plots.psychometric_plot(
        df=aud_df,
        x='col_test',
        y='first_choice_numeric',
        valueType='continue',
        bins=6,
        )

    # recompute the evidence strength
    aud_df['total_evidence_strength'] = utils.sound_evidence_strength(aud_df['percentage_of_timebins_with_evidence_high'], aud_df['percentage_of_timebins_with_evidence_low'])
    aud_df = aud_df[["subject", "session", "trial", "date", "correct", "correct_side", "current_training_stage", "difficulty",
                     "stimulus_modality", "water", "miss_trial", "first_choice", "last_choice", "early_pokeout", "trial_of_day","number_of_tones_high","number_of_tones_low"
                     , "total_percentage_of_tones_high", "total_percentage_of_tones_low", "percentage_of_timebins_with_evidence_high", "percentage_of_timebins_with_evidence_low"
                    , "total_evidence_strength", ]]
    return (aud_df,)


@app.cell
def _(df, dft, np, pd, utils):
    vis_df = df[df.current_training_stage == 'TwoAFC_visual_hard']
    vis_df = dft.get_performance_by_difficulty_ratio(vis_df)
    vis_df["visual_stimulus_ratio"] = vis_df["visual_stimulus_ratio"].round(3)
    vis_df = dft.get_left_choice(vis_df)
    # let's use the absolute value of the lowest visual stimulus as a proxy for the brightness of the visual stimulus
    vis_df['visual_stimulus_lowest'] = vis_df['visual_stimulus'].apply(lambda x: abs(eval(x)[0]) if eval(x)[0] < eval(x)[1] else abs(eval(x)[1]))
    # create 5 bins for the absolute value of the lowest visual stimulus
    min_value = vis_df['visual_stimulus_lowest'].min()
    max_value = vis_df['visual_stimulus_lowest'].max()
    bins = np.linspace(min_value, max_value, 6)
    vis_df['visual_stimulus_lowest_binned'] = pd.cut(vis_df['visual_stimulus_lowest'], bins=bins, labels=[f"{b:.2f}" for b in bins[:-1]])
    # explore reaction times
    vis_df["port2_holds"] = vis_df.apply(lambda row: utils.get_trial_port_hold(row, 2), axis=1)
    vis_df["port2_holds_number"] = vis_df["port2_holds"].apply(lambda x: len(x))
    vis_df["visual_stimulus_ratio_log"] = np.sign(vis_df["visual_stimulus_ratio"]) * (np.log(abs(vis_df["visual_stimulus_ratio"])).round(4))
    vis_df['visual_stimulus_ratio_log_abs'] = vis_df['visual_stimulus_ratio_log'].abs()
    return (vis_df,)


@app.cell
def _(aud_df):
    aud_df.to_parquet("auditory_2AFC.parquet")
    aud_df
    return


@app.cell
def _(vis_df):
    vis_df.to_parquet("visual_2AFC.parquet")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
