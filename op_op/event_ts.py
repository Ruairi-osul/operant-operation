import numpy as np
import scipy.signal
import pandas as pd


def events_to_time_series(
    events_array: np.ndarray | pd.Series,
    sampling_interval: float = 0.1,
    kernel: np.ndarray | None = None,
    total_duration: float | None = None,
    shift: int = 0,
) -> np.ndarray:
    """
    Convert an array of event times to a time series.

    Args:
        events_array (np.ndarray): Array of event times in float seconds.
        sampling_interval (float): Sampling interval of the time series in seconds.
        kernel (np.ndarray): Optional kernel to convolve the time series with.
        total_duration (float): Total duration of the time series in seconds.
        shift (int): Number of samples to shift the time series by. Can be negative for reverse shifts.

    Returns:
        np.ndarray: Time series of the events.
    """
    if total_duration is None:
        total_duration = np.max(events_array)

    n_timesteps = int(np.ceil(total_duration / sampling_interval))

    time_series = np.zeros(n_timesteps)

    event_indices = (np.array(events_array) / sampling_interval).astype(int)

    time_series[event_indices] = 1
    if kernel is not None:
        time_series = scipy.signal.convolve(time_series, kernel, mode="same")

    time_series = np.roll(time_series, shift)

    return time_series


def event_time_series_time(
    total_duration: float, sampling_interval: float = 0.1, start_time: float = 0.0
) -> np.ndarray:
    """
    Create a time series of the time points for pairing with an event time series.

    Args:
        total_duration (float): Total duration of the time series in seconds.
        sampling_interval (float): Sampling interval of the time series in seconds.
        start_time (float): Time of the first time point in seconds.

    Returns:
        np.ndarray: Time series of the time points.
    """
    n_timesteps = int(np.ceil(total_duration / sampling_interval))
    time = np.arange(n_timesteps) * sampling_interval + start_time
    return time


def events_to_time_series_df(
    events_array: np.ndarray | pd.Series,
    event_name: str,
    total_duration: float,
    num_shifts_forwards: int = 5,
    num_shifts_backwards: int = 5,
    sampling_interval: float = 0.1,
    kernel: np.ndarray | None = None,
    created_time_col: str = "time",
) -> pd.DataFrame:
    """
    Create a dataframe of time series for a set of events.

    Time series are created for the event times and shifted forwards and backwards in time using np.roll.

    Args:
        events_array (np.ndarray): Array of event times in float seconds.
        event_name (str): Name of the event. Used for column names. Created columns have the format "{event_name}_at_{shift}".
        total_duration (float): Total duration of the time series in seconds.
        num_shifts_forwards (int): Number of shifts to create forwards in time.
        num_shifts_backwards (int): Number of shifts to create backwards in time.
        sampling_interval (float): Sampling interval of the time series in seconds.
        kernel (np.ndarray): Optional kernel to convolve the time series with.
        created_time_col (str): Name of the time column.

    Returns:
        pd.DataFrame: Dataframe of time series.
    """
    shifts = np.arange(-num_shifts_backwards, num_shifts_forwards + 1)
    data_dict = {
        f"{event_name}_at_{shift}": events_to_time_series(
            events_array=events_array,
            sampling_interval=sampling_interval,
            kernel=kernel,
            total_duration=total_duration,
            shift=shift,
        )
        for shift in shifts
    }
    data_dict[created_time_col] = event_time_series_time(
        total_duration=total_duration, sampling_interval=sampling_interval
    )

    df = pd.DataFrame(data_dict)
    df = df[[created_time_col] + [c for c in df.columns if c != created_time_col]]
    return df
