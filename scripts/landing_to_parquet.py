import pandas as pd
import os
from pathlib import Path
from typing import List
import numpy as np

LANDING_DIR = Path(os.environ.get("DATA_DIR")) / os.environ.get("LANDING_PREFIX")
RAW_DIR = Path(os.environ.get("DATA_DIR")) / os.environ.get("RAW_PREFIX")
PARQUET_COMPRESSION = "snappy"
OUTPUT_EVENTS_FILENAME = "events.parquet"
OUTPUT_RAW_CA_FILENAME = "raw_calcium.parquet"
OUTPUT_DECONV_CA_FILENAME = "deconv_calcium.parquet"
OUTPUT_MOTION_FILENAME = "motion_tracking.parquet"


class FileProcessor:
    def __init__(
        self,
        output_filename: str,
        source_filename: str,
        compression: str = "snappy",
    ):
        self.compression = compression
        self.output_filename = output_filename
        self.source_filename = source_filename

    @staticmethod
    def read_landing_file(file_path: Path) -> pd.DataFrame:
        return pd.read_csv(file_path)

    @staticmethod
    def find_files(landing_dir: Path, source_filename: str) -> List[Path]:
        return list(landing_dir.rglob(f"*{source_filename}"))

    @staticmethod
    def process_file(df: Path) -> pd.DataFrame:
        return df

    @staticmethod
    def get_output_path(
        landing_dir: Path,
        raw_dir: Path,
        source_file: Path,
        output_filename: str | None = None,
    ) -> Path:
        source_file = Path(source_file).resolve()
        landing_dir = Path(landing_dir).resolve()
        raw_dir = Path(raw_dir).resolve()

        if not source_file.is_relative_to(landing_dir):
            raise ValueError("The file is not located under the source directory")

        rel_path = source_file.relative_to(landing_dir)
        output_path = raw_dir / rel_path

        if output_filename is not None:
            output_path = output_path.with_name(output_filename)

        return output_path

    def process_files(
        self,
        landing_dir: Path,
        raw_dir: Path,
    ) -> List[Path]:
        source_files = self.find_files(landing_dir, self.source_filename)
        dest_files = []

        for source_file in source_files:
            output_path = self.get_output_path(
                landing_dir=landing_dir,
                raw_dir=raw_dir,
                source_file=source_file,
                output_filename=self.output_filename,
            )

            df = self.read_landing_file(source_file)
            df = self.process_df(df)

            df.to_parquet(output_path, compression=self.compression, index=False)
            dest_files.append(output_path)
        return dest_files


class EventProcessor(FileProcessor):
    def __init__(
        self,
        compression: str = "snappy",
        source_filename: str = "events.csv",
        output_filename="events.parquet",
    ):
        super().__init__(
            compression=compression,
            source_filename=source_filename,
            output_filename=output_filename,
        )

    @staticmethod
    def process_df(df: pd.DataFrame) -> pd.DataFrame:
        cols = [
            "trial_idx",
            "block_type",
            "was_omission",
            "was_forced_choice",
            "start_time",
            "mouse_init_time",
            "screen_touch_time",
            "reward_collection_time",
            "chose_large",
            "was_shocked",
        ]

        df1 = (
            df.rename(
                columns=dict(
                    Trial="trial_idx",
                    Block="block_type",
                    omissionALL="was_omission",
                    ForceFree="was_forced_choice",
                    TrialPossible="start_time",
                    stTime="mouse_init_time",
                    choiceTime="screen_touch_time",
                    collectionTime="reward_collection_time",
                    bigSmall="chose_large",
                    shock="was_shocked",
                )
            )
            .assign(
                was_omission=lambda x: x.was_omission.astype(bool),
                was_forced_choice=lambda x: x.was_forced_choice.astype(bool),
                chose_large=lambda x: x.chose_large == 1.2,
                was_shocked=lambda x: x.was_shocked.astype(bool),
            )
            .loc[:, cols]
            .copy()
        )
        return df1


class CaProcessor(FileProcessor):
    @staticmethod
    def read_landing_file(file_path: Path) -> pd.DataFrame:
        df = pd.read_csv(file_path, header=None)
        df = df.transpose()
        return df

    @staticmethod
    def process_df(df: pd.DataFrame) -> pd.DataFrame:
        neuron_cols = [f"n{i}" for i in range(len(df.columns))]
        df.columns = neuron_cols

        df["time"] = np.arange(len(df)) * 0.1
        df = df[["time"] + neuron_cols]
        return df


class DeconvCaProcessor(CaProcessor):
    def __init__(
        self,
        compression: str = "snappy",
        source_filename: str = "deconv_calcium.csv",
        output_filename="deconv_calcium.parquet",
    ):
        super().__init__(
            compression=compression,
            source_filename=source_filename,
            output_filename=output_filename,
        )


class RawCaProcessor(DeconvCaProcessor):
    def __init__(
        self,
        compression: str = "snappy",
        source_filename: str = "raw_calcium.csv",
        output_filename="raw_calcium.parquet",
    ):
        super().__init__(
            compression=compression,
            source_filename=source_filename,
            output_filename=output_filename,
        )


class MotionProcessor(FileProcessor):
    def __init__(
        self,
        compression: str = "snappy",
        source_filename: str = "motion_tracking.csv",
        output_filename="motion_tracking.parquet",
    ):
        super().__init__(
            compression=compression,
            source_filename=source_filename,
            output_filename=output_filename,
        )

    @staticmethod
    def rename_cols(df: pd.DataFrame) -> pd.DataFrame:
        return df.rename(
            columns=dict(x_pix="x", y_pix="y", idx_time="time", idx_frame="frame_idx")
        )

    @staticmethod
    def round_rime(df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(time=lambda x: x.time.round(1))

    @staticmethod
    def add_velocity(
        df: pd.DataFrame,
        x_col: str = "x",
        y_col: str = "y",
        time_col: str = "time",
        created_velocity_col="velocity",
    ) -> pd.DataFrame:
        df_sorted = df.sort_values(by=time_col)

        dx = df_sorted[x_col].diff()
        dy = df_sorted[y_col].diff()

        dt = df_sorted[time_col].diff()

        vx = dx / dt
        vy = dy / dt

        df_sorted[created_velocity_col] = np.sqrt(vx**2 + vy**2)

        return df_sorted

    def select_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[["frame_idx", "time", "x", "y", "velocity"]]

    def process_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df1 = self.rename_cols(df)
        df1 = self.round_rime(df1)
        df2 = self.add_velocity(df1)
        df3 = self.select_cols(df2)
        return df3


def process_events(input_dir, output_dir) -> List[Path]:
    event_processor = EventProcessor(
        compression=PARQUET_COMPRESSION,
        output_filename=OUTPUT_EVENTS_FILENAME,
    )
    processed_event_paths = event_processor.process_files(
        input_dir,
        output_dir,
    )
    return processed_event_paths


def process_deconv_calcium(input_dir, output_dir) -> List[Path]:
    deconv_calcium_processor = DeconvCaProcessor(
        compression=PARQUET_COMPRESSION,
        output_filename=OUTPUT_DECONV_CA_FILENAME,
    )

    processed_deconv_calcium_paths = deconv_calcium_processor.process_files(
        input_dir,
        output_dir,
    )
    return processed_deconv_calcium_paths


def process_raw_calcium(input_dir, output_dir) -> List[Path]:
    raw_calcium_processor = RawCaProcessor(
        compression=PARQUET_COMPRESSION,
        output_filename=OUTPUT_RAW_CA_FILENAME,
    )
    processed_raw_calcium_paths = raw_calcium_processor.process_files(
        input_dir,
        output_dir,
    )
    return processed_raw_calcium_paths


def process_motion_tracking(input_dir, output_dir) -> List[Path]:
    motion_processor = MotionProcessor(
        compression=PARQUET_COMPRESSION,
        output_filename=OUTPUT_MOTION_FILENAME,
    )
    processed_motion_paths = motion_processor.process_files(
        input_dir,
        output_dir,
    )
    return processed_motion_paths


def main():
    _ = process_events(input_dir=LANDING_DIR, output_dir=RAW_DIR)
    _ = process_deconv_calcium(input_dir=LANDING_DIR, output_dir=RAW_DIR)
    _ = process_raw_calcium(input_dir=LANDING_DIR, output_dir=RAW_DIR)
    _ = process_motion_tracking(input_dir=LANDING_DIR, output_dir=RAW_DIR)


if __name__ == "__main__":
    main()
