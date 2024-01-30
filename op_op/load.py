from pathlib import Path
import pandas as pd

FILENAMES = {
    "events": "events.parquet",
    "motion": "motion_tracking.parquet",
    "deconv_calcium": "deconv_calcium.parquet",
    "raw_calcium": "raw_calcium.parquet",
}


def load_session_data(
    data_dir: Path,
    mouse_name: str,
    session: str,
    data_type: str,
    add_meta_cols: bool = False,
) -> pd.DataFrame:
    """
    Load a single data file for a given mouse and session.

    Args:
        data_dir (Path): Path to the root data directory.
        mouse_name (str): Name of the mouse. Must be a subdirectory of data_dir.
        session (str): Name of the session. Must be a subdirectory of data_dir/mouse_name.
        data_type (str): Type of data to load. Must be one of "events", "motion", "deconv_calcium", or "raw_calcium".
        add_meta_cols (bool): If True, add columns for mouse_name and session to the DataFrame.

    Returns:
        pd.DataFrame: DataFrame containing the data.

    Raises:
        ValueError: If data_type is not one of "events", "motion", "deconv_calcium", or "raw_calcium".
        FileNotFoundError: If the file does not exist.
    """
    try:
        filename = FILENAMES[data_type]
    except KeyError:
        raise ValueError(f"Unknown data type {data_type}.")

    file_path = data_dir / mouse_name / session / filename
    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist.")

    df = pd.read_parquet(file_path)
    if add_meta_cols:
        df["mouse_name"] = mouse_name
        df["session"] = session

    return df


def load_events(
    data_dir: Path, mouse_name: str, session: str, add_meta_cols: bool = False
) -> pd.DataFrame:
    """
    Load the events data for a given mouse and session.

    Args:
        data_dir (Path): Path to the root data directory.
        mouse_name (str): Name of the mouse. Must be a subdirectory of data_dir.
        session (str): Name of the session. Must be a subdirectory of data_dir/mouse_name.
        add_meta_cols (bool): If True, add columns for mouse_name and session to the DataFrame.

    Returns:
        pd.DataFrame: DataFrame containing the events data.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    return load_session_data(data_dir, mouse_name, session, "events", add_meta_cols)


def load_motion(
    data_dir: Path, mouse_name: str, session: str, add_meta_cols: bool = False
) -> pd.DataFrame:
    """
    Load the motion tracking data for a given mouse and session.

    Args:
        data_dir (Path): Path to the root data directory.
        mouse_name (str): Name of the mouse. Must be a subdirectory of data_dir.
        session (str): Name of the session. Must be a subdirectory of data_dir/mouse_name.
        add_meta_cols (bool): If True, add columns for mouse_name and session to the DataFrame.

    Returns:
        pd.DataFrame: DataFrame containing the motion tracking data.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    return load_session_data(data_dir, mouse_name, session, "motion", add_meta_cols)


def load_deconv_calcium(
    data_dir: Path, mouse_name: str, session: str, add_meta_cols: bool = False
) -> pd.DataFrame:
    """
    Load the deconvolved calcium data for a given mouse and session.

    Args:
        data_dir (Path): Path to the root data directory.
        mouse_name (str): Name of the mouse. Must be a subdirectory of data_dir.
        session (str): Name of the session. Must be a subdirectory of data_dir/mouse_name.
        add_meta_cols (bool): If True, add columns for mouse_name and session to the DataFrame.

    Returns:
        pd.DataFrame: DataFrame containing the deconvolved calcium data.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    return load_session_data(
        data_dir, mouse_name, session, "deconv_calcium", add_meta_cols
    )


def load_raw_calcium(
    data_dir: Path, mouse_name: str, session: str, add_meta_cols: bool = False
) -> pd.DataFrame:
    """
    Load the raw calcium data for a given mouse and session.

    Args:
        data_dir (Path): Path to the root data directory.
        mouse_name (str): Name of the mouse. Must be a subdirectory of data_dir.
        session (str): Name of the session. Must be a subdirectory of data_dir/mouse_name.
        add_meta_cols (bool): If True, add columns for mouse_name and session to the DataFrame.

    Returns:
        pd.DataFrame: DataFrame containing the raw calcium data.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    return load_session_data(
        data_dir, mouse_name, session, "raw_calcium", add_meta_cols
    )
