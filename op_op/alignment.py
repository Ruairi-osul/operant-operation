import pandas as pd
import numpy as np
from calcium_clear.align import align_to_events


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


def demarcate_latency_from_event(
    df_ts: pd.DataFrame,
    df_events: pd.DataFrame,
    event_time_col: str = "start_time",
    t_before_event: float = 0,
    t_after_event: float = 10100,
    df_ts_time_col: str = "time",
    created_latency_col: str | None = None,
    created_event_idx_col: str | None = None,
    copy: bool = False,
    round_precision: int = 1,
    adjust_backward: bool = True,
) -> pd.DataFrame:
    """
    Demarcates latency from event for time series such as calcium or motion data.

    Args:
        df_ts (pd.DataFrame): Time series dataframe with time column
        df_events (pd.DataFrame): Event dataframe with event time column
        event_time_col (str, optional): Event time column in df_events. Defaults to "start_time".
        t_before_event (float, optional): Time before event to demarcate. Defaults to 0.
        t_after_event (float, optional): Time after event to demarcate. Defaults to 10100.
        df_ts_time_col (str, optional): Time column in df_ts. Defaults to "time".
        created_latency_col (str, optional): Latency column name to create. Defaults to f"latency_from_{event_time_col}".
        created_event_idx_col (str, optional): Event index column name to create. Defaults to None.
        copy (bool, optional): Copy df_ts before demarcating. Defaults to False.
        round_precision (int, optional): Round precision of latency column. Defaults to 1.
        adjust_backward (bool, optional): Adjust latency backward by median difference between latencies. Set to False when aligning backwards. Defaults to True.

    Returns:
        pd.DataFrame: df_ts with latency column
    """
    created_latency_col = created_latency_col or f"latency_from_{event_time_col}"
    temp_event_idx_col = "event_idx_TEMP"

    if copy:
        df_ts = df_ts.copy()

    df_aligned = align_to_events(
        df_wide=df_ts,
        events=df_events[event_time_col],
        t_before=t_before_event,
        t_after=t_after_event,
        time_col=df_ts_time_col,
        created_event_index_col=temp_event_idx_col,
        created_aligned_time_col=created_latency_col,
        drop_non_aligned=False,
        round_precision=round_precision,
    )
    if adjust_backward:
        dt = df_aligned[created_latency_col].diff().median()
        df_aligned[created_latency_col] = df_aligned[created_latency_col] - dt
        df_aligned[created_latency_col] = df_aligned[created_latency_col].round(
            round_precision
        )
    if created_event_idx_col is None:
        df_aligned = df_aligned.drop(columns=temp_event_idx_col)
    else:
        df_aligned = df_aligned.rename(
            columns={temp_event_idx_col: created_event_idx_col}
        )

    # reorder columns
    first_cols = [df_ts_time_col, created_latency_col]
    if created_event_idx_col is not None:
        first_cols.append(temp_event_idx_col)
    other_cols = [c for c in df_aligned.columns if c not in first_cols]
    df_aligned = df_aligned[first_cols + other_cols]

    return df_aligned
