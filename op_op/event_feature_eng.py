import pandas as pd
import numpy as np


def num_prev_trials(
    df_events: pd.DataFrame, in_block: bool = False, block_type_col: str = "block_type"
) -> np.ndarray:
    """
    Returns the number of previous trials for each event, optionally since the start of the current block.

    Args:
        df_events (pd.DataFrame): A DataFrame containing event data.
        in_block (bool, optional): Whether to calculate the number of previous trials within a block. Defaults to False.

    Returns:
        np.ndarray: An array containing the number of previous trials for each event.
    """
    trial_idx = np.arange(len(df_events))

    if in_block:
        for block_type in df_events[block_type_col].unique():
            idx = df_events[block_type_col] == block_type
            trial_idx[idx] = trial_idx[idx] - trial_idx[idx].min()

    return trial_idx


def cumulative_reward(
    df_events: pd.DataFrame,
    large_choice_col: str = "chose_large",
    large_reward_amt: float = 1.0,
    small_reward_amt: float = 0.0,
) -> np.ndarray:
    """
    Returns the cumulative reward up to each event.

    Args:
        df_events (pd.DataFrame): A DataFrame containing event data.
        large_choice_col (str, optional): The column containing the large choice data. Defaults to "chose_large".
        large_reward_amt (float, optional): The reward amount for a large choice. Defaults to 1.0.
        small_reward_amt (float, optional): The reward amount for a small choice. Defaults to 0.0.
        window (int, optional): The size of the window to use for smoothing. Defaults to entire session.

    Returns:
        np.ndarray: An array containing the cumulative reward up to each event.
    """
    reward = (
        df_events[large_choice_col]
        .shift(1)
        .map({True: large_reward_amt, False: small_reward_amt, np.nan: 0.0})
    )
    return reward.cumsum()


def reward_in_window(
    df_events: pd.DataFrame,
    window_kernel: np.ndarray,
    large_choice_col: str = "chose_large",
    large_reward_amt: float = 1.0,
    small_reward_amt: float = 0.0,
) -> np.ndarray:
    """
    Returns the cumulative reward in trialing window preceding each event.

    Args:
        df_events (pd.DataFrame): A DataFrame containing event data.
        window_kernel (np.ndarray): The kernel to use for smoothing. Size determines window size.
        large_choice_col (str, optional): The column containing the large choice data. Defaults to "chose_large".
        large_reward_amt (float, optional): The reward amount for a large choice. Defaults to 1.0.
        small_reward_amt (float, optional): The reward amount for a small choice. Defaults to 0.0.

    Returns:
        np.ndarray: An array containing the cumulative reward up to each event.
    """
    rewards = np.where(df_events[large_choice_col], large_reward_amt, small_reward_amt)

    cumulative_rewards = np.convolve(rewards, window_kernel, mode="full")[
        : len(rewards)
    ]

    # shift forward and pad zero
    cumulative_rewards = np.roll(cumulative_rewards, 1)
    cumulative_rewards[0] = 0

    return cumulative_rewards


def cumulative_shock(
    df_events: pd.DataFrame,
    shock_col: str = "was_shocked",
    shock_amt: float = 1.0,
    not_shocked_amt: float = 0.0,
) -> np.ndarray:
    """
    Returns the cumulative shock up to each event.

    Args:
        df_events (pd.DataFrame): A DataFrame containing event data.
        shock_col (str, optional): The column containing the shock data. Defaults to "shock".
        shock_amt (float, optional): The shock amount. Defaults to 1.0.
        not_shocked_amt (float, optional): The amount for no shock. Defaults to 0.0.

    Returns:
        np.ndarray: An array containing the cumulative shock up to each event.
    """
    shock = np.where(
        df_events.shift(1).fillna(0)[shock_col], shock_amt, not_shocked_amt
    )
    return shock.cumsum()


def shock_in_window(
    df_events: pd.DataFrame,
    window_kernel: np.ndarray,
    shock_col: str = "was_shocked",
    shock_amt: float = 1.0,
    not_shocked_amt: float = 0.0,
) -> np.ndarray:
    """
    Returns the cumulative shock in trialing window preceding each event.

    Args:
        df_events (pd.DataFrame): A DataFrame containing event data.
        window_kernel (np.ndarray): The kernel to use for smoothing. Size determines window size.
        shock_col (str, optional): The column containing the shock data. Defaults to "shock".
        shock_amt (float, optional): The shock amount. Defaults to 1.0.
        not_shocked_amt (float, optional): The amount for no shock. Defaults to 0.0.

    Returns:
        np.ndarray: An array containing the cumulative shock up to each event.
    """
    shocks = np.where(df_events[shock_col], shock_amt, not_shocked_amt)

    cumulative_shocks = np.convolve(shocks, window_kernel, mode="full")[: len(shocks)]

    # shift forward and pad zero
    cumulative_shocks = np.roll(cumulative_shocks, 1)
    cumulative_shocks[0] = 0

    return cumulative_shocks


def prev_iti_length(
    df_events: pd.DataFrame,
    trial_idx_col: str = "trial_idx",
    trial_start_col: str = "start_time",
    trial_env_col: str = "reward_collection_time",
) -> np.ndarray:
    """
    Returns the duration of the inter-trial interval (ITI) preceding each event.

    Args:
        df_events (pd.DataFrame): A DataFrame containing event data.
        trial_idx_col (str, optional): The column containing the trial index data. Defaults to "trial_idx".
        trial_start_col (str, optional): The column containing the trial start time data. Defaults to "start_time".
        trial_env_col (str, optional): The column containing the trial end time data. Defaults to "reward_collection_time".

    Returns:
        np.ndarray: An array containing the duration of the ITI preceding each event.
    """
    iti = np.zeros(len(df_events))
    iti[0] = np.nan
    for i, trial_idx in enumerate(df_events[trial_idx_col].unique()):
        if i == 0:
            continue

        idx_current_trial = df_events[trial_idx_col] == trial_idx
        idx_prev_trial = df_events[trial_idx_col] == trial_idx - 1
        start_time = df_events.loc[idx_current_trial, trial_start_col].values[0]
        end_time = df_events.loc[idx_prev_trial, trial_env_col].values[0]

        iti[idx_current_trial] = start_time - end_time

    return iti


def event_delay(
    df_events: pd.DataFrame,
    first_event_col: str,
    second_event_col: str,
    trial_idx_col: str = "trial_idx",
    shift: int = 1,
) -> np.ndarray:
    """
    Calculate the delay between a two events occuring on the same trial.

    Args:
        df_events (pd.DataFrame): A DataFrame containing event data.
        first_event_col (str): The column containing the first event data.
        second_event_col (str): The column containing the second event data.
        trial_idx_col (str, optional): The column containing the trial index data. Defaults to "trial_idx".
        shift (int, optional): Trial shift. Current trial if 0, previous trial if 1. Defaults to 1.

    Returns:
        np.ndarray: An array containing the delay between the two events.
    """
    df = df_events.sort_values(trial_idx_col).shift(shift)

    delay = df[second_event_col] - df[first_event_col]

    return delay.values


def trial_start_delay(
    df_events: pd.DataFrame,
    trial_idx_col: str = "trial_idx",
    trial_start_col: str = "start_time",
    mouse_init_col: str = "mouse_init_time",
    shift: int = 1,
) -> np.ndarray:
    """
    Calculate the delay between a trial start and mouse initiation.

    Args:
        df_events (pd.DataFrame): A DataFrame containing event data.
        trial_idx_col (str, optional): The column containing the trial index data. Defaults to "trial_idx".
        trial_start_col (str, optional): The column containing the trial start time data. Defaults to "start_time".
        mouse_init_col (str, optional): The column containing the mouse initiation time data. Defaults to "mouse_init_time".
        shift (int, optional): Trial shift. Current trial if 0, previous trial if 1. Defaults to 1.

    Returns:
        np.ndarray: An array containing the delay between a trial start and mouse initiation.
    """
    delay = event_delay(
        df_events,
        first_event_col=trial_start_col,
        second_event_col=mouse_init_col,
        trial_idx_col=trial_idx_col,
        shift=shift,
    )
    return delay


def screen_touch_delay(
    df_events: pd.DataFrame,
    trial_idx_col: str = "trial_idx",
    mouse_init_col: str = "mouse_init_time",
    screen_touch_col: str = "screen_touch_time",
    shift: int = 1,
) -> np.ndarray:
    """
    Calculate the delay between mouse initiation and screen touch.

    Args:
        df_events (pd.DataFrame): A DataFrame containing event data.
        trial_idx_col (str, optional): The column containing the trial index data. Defaults to "trial_idx".
        mouse_init_col (str, optional): The column containing the mouse initiation time data. Defaults to "mouse_init_time".
        screen_touch_col (str, optional): The column containing the screen touch time data. Defaults to "screen_touch_time".
        shift (int, optional): Trial shift. Current trial if 0, previous trial if 1. Defaults to 1.

    Returns:
        np.ndarray: An array containing the delay between mouse initiation and screen touch.
    """
    delay = event_delay(
        df_events,
        first_event_col=mouse_init_col,
        second_event_col=screen_touch_col,
        trial_idx_col=trial_idx_col,
        shift=shift,
    )
    return delay


def reward_collection_delay(
    df_events: pd.DataFrame,
    trial_idx_col: str = "trial_idx",
    screen_touch_col: str = "screen_touch_time",
    reward_col: str = "reward_collection_time",
    shift: int = 1,
) -> np.ndarray:
    """
    Calculate the delay between screen touch and reward collection.

    Args:
        df_events (pd.DataFrame): A DataFrame containing event data.
        trial_idx_col (str, optional): The column containing the trial index data. Defaults to "trial_idx".
        screen_touch_col (str, optional): The column containing the screen touch time data. Defaults to "screen_touch_time".
        reward_col (str, optional): The column containing the reward collection time data. Defaults to "reward_collection_time".
        shift (int, optional): Trial shift. Current trial if 0, previous trial if 1. Defaults to 1.

    Returns:
        np.ndarray: An array containing the delay between screen touch and reward collection.
    """
    delay = event_delay(
        df_events,
        first_event_col=screen_touch_col,
        second_event_col=reward_col,
        trial_idx_col=trial_idx_col,
        shift=shift,
    )
    return delay


def trial_duration(
    df_events: pd.DataFrame,
    trial_idx_col: str = "trial_idx",
    trial_start_col: str = "start_time",
    reward_col: str = "reward_collection_time",
    shift: int = 1,
) -> np.ndarray:
    """
    Calculate the duration of a trial.

    Args:
        df_events (pd.DataFrame): A DataFrame containing event data.
        trial_idx_col (str, optional): The column containing the trial index data. Defaults to "trial_idx".
        trial_start_col (str, optional): The column containing the trial start time data. Defaults to "start_time".
        reward_col (str, optional): The column containing the reward collection time data. Defaults to "reward_collection_time".
        shift (int, optional): Trial shift. Current trial if 0, previous trial if 1. Defaults to 1.

    Returns:
        np.ndarray: An array containing the duration of a trial.
    """
    delay = event_delay(
        df_events,
        first_event_col=trial_start_col,
        second_event_col=reward_col,
        trial_idx_col=trial_idx_col,
        shift=shift,
    )
    return delay
