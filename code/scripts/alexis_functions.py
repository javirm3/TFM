# Import libraries (need to reduce scipy's entries)
import numpy as np
from scipy.signal import firwin, lfilter  # For white_noise
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
# from sympy import symbols, Eq, log, nsolve  # Not installed in setup PCs
from scipy.optimize import fsolve
# import slack
import os
import csv
import random
from matplotlib import pyplot as plt
from pathlib import Path

# For compute_psych_curve
from scipy import stats
from scipy.optimize import minimize
from collections import namedtuple
import time
from functools import wraps
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths

# For saving all figures in a notebook
import nbformat
import base64


def power_dB(amp):
    """Transform amplitude into decibels (dB)"""
    amp_ref = 0.00002  # The commonly used reference sound pressure in air is 20 µPa
    dB = 20 * np.log10(amp / amp_ref)
    # x, y = find_power_dB_par()
    # dB = 15.535 * np.log10((amp + 0.00267) / amp_ref)
    # dB = x * np.log10((amp + y) / amp_ref)
    return dB


def ild():
    """Get the inter aural level difference (ild) of a sound given its evidence (-1=left, 1=right).
    The input should be a csv file to convert to DataFrame. Only for sounds.csv (batch 1)
    """
    path = paths.ALEXIS / 'sounds.csv'  # My laptop
    df = pd.read_csv(path)
    # behavior_filenames = pd.read_csv(path).drop('filename', 1)  # Import csv as DataFrame dropping the column 'filename'
    df_dB = df  # Copy DataFrame
    df_dB.iloc[:, 1:21] = power_dB(abs(df.iloc[:, 1:21]))  # Apply the function to entire DataFrame except 'filename'
    # column. abs because can't do log10 of negative number. To retrieve the negative sign for left the ILD will be
    # computed as right - left later
    df_dB_left = df_dB.iloc[:, 1:11]  # Index left skipping 'filename'
    df_dB_left.columns = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']  # DataFrames needs to have BOTH the same
    # row and column indices in order to perform an element-wise subtraction
    df_dB_right = df_dB.iloc[:, 11:21]  # Index right
    df_dB_right.columns = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    df_ild = df_dB_right - df_dB_left  # Interaural level difference. Right minus left so we get negative for left and
    # positive for right
    df_ild.columns = ['ILD0', 'ILD1', 'ILD2', 'ILD3', 'ILD4', 'ILD5', 'ILD6', 'ILD7', 'ILD8',
                      'ILD9']  # Change column labels
    df_ild['Mean'] = df_ild.mean(axis=1)  # Add mean ILD per sound at the end of the DataFrame
    df_ild.insert(0, 'Filename', df.filename)  # Insert in 'filename' i the first column
    evidences = np.array([-1, -0.9, -0.8, -0.75, -0.6, -0.5, -0.4, -0.3, -0.25, -0.1,
                          0, 0.1, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 0.8, 0.9, 1])  # Define evidences
    df_ild.insert(1, 'Evidence',
                  np.repeat(evidences, len(evidences) ** 2))  # Repeat each evidence per n sounds with that evidence
    # df_ild_summary = behavior_filenames.groupby('Evidence', as_index=False).mean()  # SQL-style index
    df_ild_summary = df_ild.groupby('Evidence').mean()  # Group labels asn index
    return df_ild


def get_ild(stim_set=6):
    # Load sounds
    if stim_set == 1:
        sounds_path = paths.ALEXIS / 'sounds.csv'
    if stim_set == 2:
        sounds_path = paths.ALEXIS / 'sounds_2.csv'
    elif stim_set == 6:
        sounds_path = paths.ALEXIS / 'sounds_6.1.csv'

    sounds = pd.read_csv(sounds_path)
    n_frames = sounds.n_frames.unique()[0]

    # Left frames
    left_frames_column_names = [f'EL{n:01}' for n in range(n_frames)]
    frames_left = sounds[left_frames_column_names].values

    # Right frames
    right_frames_column_names = [f'ER{n:01}' for n in range(n_frames)]
    frames_right = sounds[right_frames_column_names].values

    # Frames ILD (elementwise substraction)
    frames_ild = frames_right - frames_left
    frames_ild = pd.DataFrame(frames_ild)
    frames_ild.insert(0, column='filename', value=sounds.filename)  # Insert behavior_filenames in first column

    return frames_ild




def compute_psych_curve(x, y, n_points=100):
    """Computes a psychometric function."""

    # https://psychology.stackexchange.com/questions/13347/how-can-i-fit-a-psychometric-function-such-that-the-minimum-is-50-chance-level

    def sigmoid_mme(fit_params: tuple, xdata, ydata):
        k, x0, b, p = fit_params

        # k = weight (slope)
        # x0 = bias
        # b = lower lapse
        # p = upper lapse

        # Function to fit:
        y_pred = b + (1 - b - p) / (1 + np.exp(-k * (xdata - x0)))

        # Calculate negative log likelihood:
        ll = -np.sum(stats.norm.logpdf(ydata, loc=y_pred))

        return ll

    coherence_dataframe = pd.DataFrame({'r_resp': y, 'evidence': x})

    info = coherence_dataframe.groupby(['evidence'])['r_resp'].mean()
    ydata = [np.around(elem, 3) for elem in info.values]
    xdata = info.index.values
    fit_error = [np.around(elem, 3) for elem in coherence_dataframe.groupby(['evidence'])['r_resp'].sem().values]

    initial_guess = np.array([1, 1, 0, 0])

    # Run the minimizer:
    ll = minimize(sigmoid_mme, initial_guess, args=(xdata, ydata))

    # Fit parameters:
    k, x0, b, p = [np.around(param, 2) for param in ll.x]

    # Compute the fit with n_points number of points:
    fit = b + (1 - b - p) / (1 + np.exp(-k * (np.linspace(np.min(x), np.max(x), n_points) - x0)))
    fit = [np.around(elem, 3) for elem in fit]

    psych_curve = namedtuple('psych_curve',
                             ['xdata',
                              'ydata',
                              'fit',
                              'params',
                              'fit_error'])

    if len(ydata) == 0:
        return psych_curve(xdata=[np.nan],
                           ydata=[np.nan],
                           fit=[np.nan] * n_points,
                           params=[np.nan] * 4,
                           fit_error=[np.nan])
    else:
        return psych_curve(xdata=xdata,
                           ydata=ydata,
                           fit=fit,
                           params=[k, x0, b, p],
                           fit_error=fit_error)


def get_experiment(experiment=None, path_session='glue_sessions'):
    """
    Get experiment
    :param experiment: If not None, experiment=experiment. Else, show possible experiments and ask for user input.
    :param path_session: if glue_sessions look for individual sessions, elif intersession look for intersessions
    :return: experiment, path_experiment: experiment (user input), path to the experiment folder
    """

    if experiment is None:
        path_experiment = paths.ALEXIS  # Where the data for all animals is
        experiments = list(path_experiment.iterdir())  # List experiments
        experiments.sort()  # Sort them by name
        experiments = [x.name for x in path_experiment.iterdir() if x.is_dir()]  # Get rid of non folders

        try:
            experiments.remove('__pycache__')  # Pycharm's file
        except ValueError:
            pass

        print('Experiments:\n ' + str(experiments)[1:-1])  # Remove square brackets
        experiment = input('Enter experiment name')
        path_experiment = paths.ALEXIS / experiment  # Where the data for the animal is

    else:
        path_experiment = paths.ALEXIS / experiment

    return experiment, path_experiment

def get_action_trace(df, max_trial_lag=10, tau_choice=1.58, tau_error=2.22, tau_correct=0.95):
    """
    Computes the action trace for each trial in the DataFrame. The action trace is an exponentially weighted sum of past
    choices, where more recent choices have a greater influence. The weights decay exponentially with a time constant
    tau. Output is normalized between -1 and +1.
    :param df: DataFrame containing the data with a column of interest (0=left; 1=right)
    :param max_trial_lag: Number of past trials to consider
    :param tau_choice: Decay constant for choice action trace. Fitted from data (mean across subjects)
    :param tau_error: Decay constant for error action trace. Fitted from data (mean across subjects)
    :param tau_correct: Decay constant for correct action trace. Fitted from data (mean across subjects)
    :return: List of action trace values for each trial for choices, errors and correct trials
    """

    lags = np.arange(1, max_trial_lag + 1)  # Lags from 1 to k

    # Exponential decay weights
    weights_choice = np.exp(-lags / tau_choice)
    weights_error = np.exp(-lags / tau_error)
    weights_correct = np.exp(-lags / tau_correct)

    # Fixed normalizers
    Z_choice = np.sum(weights_choice)
    Z_error = np.sum(weights_error)
    Z_correct = np.sum(weights_correct)

    # Precompute signed choice
    signed_choice = 2 * df['Choice'].to_numpy() - 1  # Map 0→-1, 1→+1
    r_minus = signed_choice * df['Punish'].to_numpy()
    r_plus = signed_choice * df['Hit'].to_numpy()

    at_choice = []  # Action trace for choices
    at_error = []  # Action trace for error choices
    at_correct = []  # Action trace for correct choices

    for t in range(len(df)):
        past_choice = signed_choice[max(0, t - max_trial_lag):t]
        past_rminus = r_minus[max(0, t - max_trial_lag):t]
        past_rplus = r_plus[max(0, t - max_trial_lag):t]

        # Slice the weights to match available history and reverse
        w_choice = weights_choice[:len(past_choice)][::-1]
        w_error = weights_error[:len(past_rminus)][::-1]
        w_correct = weights_correct[:len(past_rplus)][::-1]

        at_choice.append(np.sum(past_choice * w_choice) / Z_choice)
        at_error.append(np.sum(past_rminus * w_error) / Z_error)
        at_correct.append(np.sum(past_rplus * w_correct) / Z_correct)

    return at_choice, at_error, at_correct

def make_session_index_dm(df, column='Date'):
    """
    # Make a design matrix in which there are as many columns as unique dates. Then, for each column, there is a 1 if
    the trial belongs to that session and a 0 otherwise
    :param df: Input DataFrame
    :param column: Column of the DataFrame that contains the dates
    :return: Design matrix
    """
    dates = df[column].unique()
    design_matrix = np.zeros((len(df), len(dates)), dtype=int)
    for i, date in enumerate(dates):
        design_matrix[df[column] == date, i] = 1
    design_matrix = pd.DataFrame(design_matrix)
    return design_matrix

def make_frames_dm(df, stim_set=6, residuals=True, zscore=False):

    # Load sounds
    if stim_set == 1:
        sounds_path = paths.ALEXIS / 'sounds.csv'
    if stim_set == 2:
        sounds_path = paths.ALEXIS / 'sounds_2.csv'
    elif stim_set == 6:
        sounds_path = paths.ALEXIS / 'sounds_6.1.csv'

    sounds = pd.read_csv(sounds_path)
    n_frames = sounds.n_frames.unique()[0]
    frames_ild = get_ild(stim_set=stim_set)

    # Residuals (https://www-nature-com.sire.ub.edu/articles/nature08275)
    if residuals:
        sounds_ild = sounds.ILD
        first_frame = frames_ild[0]
        first_frame = first_frame.copy()
        first_frame.iloc[0] = 0  # Set to 0 to avoid artifact of net ILD 70 having 0 weight
        first_frame.iloc[-1] = 0  # Set to 0 to avoid artifact of net ILD 70 having 0 weight
        if stim_set == 6:
            frames_ild = frames_ild.drop(['filename', 0], axis=1).sub(sounds_ild, axis='rows')
            frames_ild.insert(0, column='filename', value=sounds.filename)  # Insert back filenames in 1st column
            frames_ild.insert(1, column=0, value=first_frame)  # Insert back first_frame in 2nd column
        else:
            frames_ild = frames_ild.drop('filename', axis=1).sub(sounds_ild, axis='rows')
            frames_ild.insert(0, column='filename', value=sounds.filename)  # Insert back filenames in 1st column

    filenames = df.Filename.tolist()

    # Get frames per trial
    stim_strength = frames_ild.loc[
        [np.where(sounds.filename == np.array(filenames[i]))[0][0] for i in range(len(filenames))]].drop(
        columns=['filename'])
    stim_strength.reset_index(drop=True, inplace=True)  # Indices must match for modeling

    # Zscore
    if not residuals:  # To not do both (otherwise I'd be subtracting the mean twice)
        if zscore:
            stim_strength = pd.DataFrame(stats.zscore(stim_strength, axis=0))  # Z-score the ILDs (along axis 0 or None
            # returns same result, but not axis 1). 0 along trials that's what I want to do :)

    design_matrix = stim_strength

    return design_matrix, n_frames

def make_net_ild_dm(df):
    """
    Make a design matrix with the net ILDs. There is a column for each absolute, unique ILD value (except 0). It
    transforms the nominal ILD into ILD net magnitude (2, 4, 8 , 70 dB) that take values +1, 0 or -1. In each trial,
    only one of these regressors is non-zero.
    When separating the stimuli S_k =  nominal_ILD + residuals and give a separate beta for the nominal and for each
    residual frame, you are somehow assuming that the impact of the nominal ILD grows linearly with ILD. But this is
    probably not the case. Particularly if spanning a range from ILD 0 to 70 dB. One simple way to not assume anything
    about how the impact of the stimuli grows with ILD is to define separate regressors for each absolute value of the
    ILD, that is 2, 4, 8 and 70. Each of this ILDs will define a regressor e.g. ILD_8 =  +1 (if ILD was +8 dB), -1 (if
    ILD was -8 db) and 0 (if ILD was other than +- 8dB). This way, you should be able to include ALL stimuli in the
    analysis (maximum evidence too).
    :param df: Input DataFrame
    :return: Design matrix
    """

    ilds = df.ILD.astype('int')
    net_ilds = np.sort(df.ILD.abs().unique().astype('int'))[1:]
    design_matrix = np.zeros((len(df), len(net_ilds)), dtype=int)
    # columns = [str(_) for _ in net_ilds]
    design_matrix = pd.DataFrame(design_matrix, columns=net_ilds)
    for i, ild in enumerate(ilds):
        if ild != 0:
            col = int(abs(ild))
            design_matrix.loc[i, col] = np.sign(ild)
    return design_matrix


def parse_glmhmm(df, covariates=None):
    """
    Parse the data for GLM-HMM with flexible covariates.
    :param df: DataFrame containing the data (one or many sessions concatenated)
    :param covariates: List of covariates to include. Options:
        'stim_vals',
        'stim_strength',
        'net_ild',
        'action_trace',
        'action_trace_error',
        'action_trace_correct',
        'bias',
        'session_index'
    :return: inputs, choices
    """

    accepted_covariates = ['stim_vals', 'stim_strength', 'net_ild', 'at_choice', 'at_error', 'at_correct', 'bias',
                           'session_index', 'prev_choice', 'wsls']
    if covariates is None:
        covariates = ['stim_vals', 'bias', 'at_choice']  # Default model
    else:
        for cov in covariates:
            if cov not in accepted_covariates:
                raise ValueError(f'Covariate {cov} not recognized. Accepted covariates are: {accepted_covariates}')

    df.reset_index(drop=True)  # Reset index to ensure consistent slicing
    dm_session_index = make_session_index_dm(df)  # Add bias (constant) per session

    # Set stimuli set
    experiment = df.Experiment.unique()[0]
    if experiment == '2AFC_6':
        stim_set = 6
    else:
        stim_set = 2

    # inputs and choices must be lists of arrays, one per session
    inputs = []
    choices = []

    for session_id, df_session in df.groupby('Session'):

        n_trials = len(df_session)
        session_cols = []

        if 'stim_vals' in covariates:
            stim_vals = df_session.ILD.values
            stim_vals = stim_vals / abs(df.ILD.max())  # Normalize ILD to [-1, 1]
            session_cols.append(stim_vals)

        if 'stim_strength' in covariates:
            stim_strength, n_frames = make_frames_dm(df_session, stim_set=stim_set, residuals=True, zscore=False)
            stim_strength = stim_strength / stim_strength.values.max()  # Normalize ILD to [-1, 1]
            session_cols.append(stim_strength)

        if 'net_ild' in covariates:
            dm_net_ild = make_net_ild_dm(df_session)
            session_cols.append(dm_net_ild)

        if 'bias' in covariates:
            session_cols.append(np.ones(n_trials))

        if 'session_index' in covariates:
            dm_session_index_sess = dm_session_index.iloc[df_session.index.values, :]
            session_cols.append(dm_session_index_sess)

        if any(x in covariates for x in ['at_choice', 'at_error', 'at_correct']):
            at_choice, at_error, at_correct = get_action_trace(df_session)
            if 'at_choice' in covariates:
                session_cols.append(np.array(at_choice))
            if 'at_error' in covariates:
                session_cols.append(np.array(at_error))
            if 'at_correct' in covariates:
                session_cols.append(np.array(at_correct))

        if 'prev_choice' in covariates:
            prev_choice = df_session.Choice.shift(1).fillna(0).values
            session_cols.append(prev_choice)

        if 'wsls' in covariates:
            wsls = df_session.Side.shift(1).fillna(0).replace({0: -1, 1: 1}).values
            session_cols.append(wsls)

        # Combine selected covariates
        session_input = np.column_stack(session_cols)
        session_choices = df_session.Choice.values.astype(int)[:, None]

        inputs.append(session_input)
        input_dim = inputs[0].shape[1]
        assert all(sess.shape[1] == input_dim for sess in inputs), 'Not all sessions have the same number of inputs'
        choices.append(session_choices)

    return inputs, choices

def clean_session_start(df):
    """
    Remove AW and WarmUp trials from one or multiple sessions.
    :param df: DataFrame containing one or more sessions
    :return: Cleaned DataFrame
    """
    def _clean(group):
        aw = group['AW'].unique()[0].astype(int)
        warmup = group['WarmUp'].unique()[0].astype(int)

        warmup_len = 40
        if warmup == 1:
            cleaned = group.iloc[warmup_len:]
        else:
            cleaned = group.copy()

        if warmup == 0 and aw > 0:
            cleaned = cleaned.iloc[aw:]

        return cleaned

    # Remove the AW and Warm Up trials
    cleaned_groups = [_clean(group) for _, group in df.groupby('Session', sort=False)]
    df_clean = pd.concat(cleaned_groups, ignore_index=True) if cleaned_groups else df
    print(f'Removed {(len(df) - len(df_clean))} trials from session start (AW and Warm Up trials)')
    return df_clean

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

    elif experiment == '2AFC_5':  # Ephys group
        df = df[df.Task == 'FD'].reset_index(drop=True)
        df = df[df.StimDur == 0.5].reset_index(drop=True)
        df = df[df.Delay == 0.5].reset_index(drop=True)
        df = df[df.P == 0].reset_index(drop=True)

    print(f'Total:{round((_ - len(df)) / 1000)}k trials')
    return df
