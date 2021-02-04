from ast import literal_eval
import numpy as np
import os
import pandas as pd
from pathlib import Path


def parse_table(path):
    pass




def add_count_data(df, col, frames_col='frames'): 
    # because frames is what the col t-min - t-max col is called
    if isinstance(df.loc[0, col], str):
        # this and the following isinstance lines are a tad fragile
        # will address this at a later date
        lens_p = [len(list(set(literal_eval(l)))) \
                  for l in df[col].values]
    else:
        m = 'Incorrect type for counting false positives'
        assert isinstance(df.loc[0, col], list), m
        lens_p = [len(list(set(l))) \
                  for l in df[col].values]
    df[col + '_count'] = lens_p
    df[col + '_per_frame'] = df[col + '_count'] / df[frames_col]
    return df


def generate_summary_df(dfs):
    col_names = [df.columns.values for df in dfs]
    #summary_cols = np.concatenate(col_names)
    means = [df.mean() for df in dfs]
    summary_cols = np.unique(np.concatenate([m.index.values for m in means]))
    sems = [df.sem() for df in dfs]
    rows = range(len(means))
    summary_df = {}
    for col in summary_cols:
        m_values = []
        s_values = []
        for i in range(len(rows)):
            if col in col_names[i]:
                new_m = means[i][col]
                new_s = sems[i][col]
            else:
                new_m, new_s = np.NaN, np.NaN
            m_values.append(new_m)
            s_values.append(new_s)
        summary_df[col + '_mean'] = m_values
        summary_df[col + '_sem'] = s_values
    summary_df = pd.DataFrame(summary_df)
    return summary_df






if __name__ == "__main__":
    from _parser import custom_parser, get_paths, track_view_base
    # -------------
    # Btrack tracks
    # -------------
    d = '/Users/amcg0011/Data/pia-tracking/200519_IVMTR69_Inj4_dmso_exp3_btrack-tracks'
    fp_n = '210122_200519_IVMTR69_Inj4_dmso_exp3_btrack-tracks_0_FP_annotations.csv'
    # false positives (swap between objects)
    fp_p = os.path.join(d, fp_n)
    fp = pd.read_csv(fp_p)
    fp['t_range'] = fp['t_stop'] - fp['t_start']
    fp = add_count_data(fp, 'FP', frames_col='t_range')
    fp = add_count_data(fp, 'FN', frames_col='t_range')
    # track terminations
    tt_n = '210122_200519_IVMTR69_Inj4_dmso_exp3_btrack-tracks_0_TT_terminations_annotations.csv'
    tt_p = os.path.join(d, tt_n)
    tt = pd.read_csv(tt_p)
    tt['t_range'] = tt['t_stop'] - tt['t_start']
    tt = add_count_data(tt, 'FP', frames_col='t_range')
    tt = add_count_data(tt, 'FN', frames_col='t_range')
    # untracked objects
    fn_n = '210131_200519_IVMTR69_Inj4_dmso_exp3_btrack-tracks_0_FN.csv'
    fn_p = os.path.join(d, fn_n)
    fn = pd.read_csv(fn_p)
    fn_m = fn.mean()
    # --------------
    # tracypy tracks
    # --------------
    # false positives
    d0 = '/Users/amcg0011/Data/pia-tracking/200519_IVMTR69_Inj4_dmso_exp3_tracks'
    fp0_n = '210123_200519_IVMTR69_Inj4_dmso_exp3_tracks_0_FP_annotations.csv'
    fp0_p = os.path.join(d0, fp0_n)
    fp0 = pd.read_csv(fp0_p)
    fp0['t_range'] = fp0['t_stop'] - fp0['t_start']
    fp0 = add_count_data(fp0, 'FP', frames_col='t_range')
    fp0 = add_count_data(fp0, 'FN', frames_col='t_range')
    # track terminations
    tt0_n = '210123_200519_IVMTR69_Inj4_dmso_exp3_tracks_0_TT_terminations_annotations.csv'
    tt0_p = os.path.join(d0, tt0_n)
    tt0 = pd.read_csv(tt0_p)
    tt0['t_range'] = tt0['t_stop'] - tt0['t_start']
    tt0 = add_count_data(tt0, 'FP', frames_col='t_range')
    tt0 = add_count_data(tt0, 'FN', frames_col='t_range')
    # untracked objects
    fn0_n = '210131_200519_IVMTR69_Inj4_dmso_exp3_tracks_0_FN.csv'
    fn0_p = os.path.join(d0, fn0_n)
    fn0 = pd.read_csv(fn0_p)
    fn0_m = fn0.mean()
    # -------------
    # Summary Stats
    # -------------
    dfs = [fp, tt, fp0, tt0]
    names = ['btrack_FP', 'btrack_TT', 'trackpy_FP', 'trackpy_TT']
    summary = generate_summary_df(dfs)
    summary['data'] = names
    # ------
    # Charts
    # ------
    import matplotlib.pyplot as plt
    import matplotlib
    x = ['BayesianTracker', 'TrackPy']
    x_pos = [0, 1]
    # Object swaps
    y = [summary.loc[0, 'FP_per_frame_mean'], summary.loc[2, 'FP_per_frame_mean']]
    sem = [summary.loc[0, 'FP_per_frame_sem'], summary.loc[2, 'FP_per_frame_sem']]
    plt.bar(x_pos, y, color='orange', yerr=sem)
    plt.xlabel('Software', size=20)
    plt.ylabel('False positive events per frame', size=20)
    matplotlib.rc('xtick', labelsize=16) 
    matplotlib.rc('ytick', labelsize=16) 
    plt.title('Object ID swaps in sampled tracks', size=20)
    plt.xticks(x_pos, x)
    plt.show()
    # Track termination rate
    y = [summary.loc[0, 'FN_per_frame_mean'], summary.loc[2, 'FN_per_frame_mean']]
    sem = [summary.loc[0, 'FN_per_frame_sem'], summary.loc[2, 'FN_per_frame_sem']]
    plt.bar(x_pos, y, color='orange', yerr=sem)
    plt.xlabel('Software', size=20)
    plt.ylabel('False negative events per frame', size=20)
    matplotlib.rc('xtick', labelsize=16) 
    matplotlib.rc('ytick', labelsize=16) 
    plt.title('Terminations observed in sampled tracks', size=20)
    plt.xticks(x_pos, x)
    plt.show()
    # Correct track termination
    y = [summary.loc[1, 'FN_count_mean'], summary.loc[3, 'FN_count_mean']]
    sem = [summary.loc[1, 'FN_count_sem'], summary.loc[3, 'FN_count_sem']]
    plt.bar(x_pos, y, color='orange', yerr=sem)
    plt.xlabel('Software', size=20)
    plt.ylabel('False terminations per sample', size=20)
    matplotlib.rc('xtick', labelsize=16) 
    matplotlib.rc('ytick', labelsize=16) 
    plt.title('False terminations per terminating track', size=20)
    plt.xticks(x_pos, x)
    plt.show()
    # Correct untracked object
    y = [fn_m['correct'] * 100, fn0_m['correct'] * 100]
    plt.bar(x_pos, y, color='orange')
    plt.xlabel('Software', size=20)
    plt.ylabel('Correctly untracked objects (%)', size=20)
    matplotlib.rc('xtick', labelsize=16) 
    matplotlib.rc('ytick', labelsize=16) 
    plt.title('Correctly untracked detected objects', size=20)
    plt.xticks(x_pos, x)
    plt.show()

