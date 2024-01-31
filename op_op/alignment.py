import pandas as pd
import numpy as np

def demarcate_trials(
    df_ts: pd.DataFrame,
    df_trials: pd.DataFrame,
    df_ts_time_col: str = "time",
    df_trials_start_col: str = "start_time",
    df_trials_end_col: str = "reward_collection_time",
    df_trials_idx_col: str = "trial_idx",
    created_trial_idx_col: str | None = None,
) -> pd.DataFrame:
    """
    Demarcates trials for time series such as calcium or motion data.

    Args:
        df_ts (pd.DataFrame): Time series dataframe with time column
        df_trials (pd.DataFrame): Trial dataframe with start and end time columns
        df_ts_time_col (str, optional): Time column in df_ts. Defaults to "time".
        df_trials_start_col (str, optional): Start time column in df_trials. Defaults to "start_time".
        df_trials_end_col (str, optional): End time column in df_trials. Defaults to "reward_collection_time".
        df_trials_idx_col (str, optional): Trial index column in df_trials. Defaults to value of "trial_idx".

    Returns:
        pd.DataFrame: df_ts with trial_id column
    """
    created_trial_idx_col = created_trial_idx_col or df_trials_idx_col

    df_ts = df_ts.sort_values(by=df_ts_time_col)
    df_trials = df_trials.sort_values(by=df_trials_start_col)

    # Creates a df with trial_id column demarcing all rows from current current trial start to next trial start with trial idx
    merged_start = pd.merge_asof(
        df_ts,
        df_trials[[df_trials_start_col, df_trials_idx_col]],
        left_on=df_ts_time_col,
        right_on=df_trials_start_col,
        direction="backward",
    )

    # Creates a df with trial_id column demarcing all rows from current current end to previous trial start with trial idx
    max_time = max(df_ts[df_ts_time_col].max(), df_trials[df_trials_end_col].max())
    df_ts["inverted_time"] = max_time - df_ts[df_ts_time_col]
    df_trials["inverted_end_time"] = max_time - df_trials[df_trials_end_col]
    merged_stop = pd.merge_asof(
        df_ts.iloc[::-1],
        df_trials[[df_trials_idx_col, "inverted_end_time"]].sort_values(
            by="inverted_end_time"
        ),
        left_on="inverted_time",
        right_on="inverted_end_time",
        direction="backward",
    )
    merged_stop = merged_stop.sort_values(by=df_ts_time_col).reset_index(drop=True)

    # Create final trial_id column by taking rows where trial_id is the same in both dfs
    df_ts[created_trial_idx_col] = merged_start[df_trials_idx_col].where(
        merged_start[df_trials_idx_col] == merged_stop[df_trials_idx_col], np.nan
    )

    # drop the temperary time columns
    df_ts.drop(columns=["inverted_time"], inplace=True)
    df_trials.drop(columns=["inverted_end_time"], inplace=True)

    # place time and trial_id columns at the front of the dataframe
    first_cols = [df_ts_time_col, created_trial_idx_col]
    other_cols = [
        col
        for col in df_ts.columns
        if col not in [df_ts_time_col, created_trial_idx_col]
    ]
    df_ts = df_ts[first_cols + other_cols]

    return df_ts
