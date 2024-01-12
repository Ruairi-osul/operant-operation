import pandas as pd
import os
from pathlib import Path

LANDING_DIR = Path(os.environ.get("DATA_DIR")) / os.environ.get("LANDING_PREFIX")
RAW_DIR = Path(os.environ.get("DATA_DIR")) / os.environ.get("RAW_PREFIX")


def convert_csv_to_parquet(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    for csv_file in input_dir.rglob("*.csv"):
        # Construct relative path for the new file in the output directory
        relative_path = csv_file.relative_to(input_dir)
        output_file = output_dir / relative_path.with_suffix(".parquet")

        # Create directories if they don't exist
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Read CSV and save as Parquet
        df = pd.read_csv(csv_file)
        df.to_parquet(output_file)


def main():
    convert_csv_to_parquet(input_dir=LANDING_DIR, output_dir=RAW_DIR)


if __name__ == "__main__":
    main()
