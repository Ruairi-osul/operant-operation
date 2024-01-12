import os
import boto3
from botocore.exceptions import NoCredentialsError
from pathlib import Path

DATA_DIR = os.environ.get("DATA_DIR", "data")
LANDING_PREFIX = os.environ.get("LANDING_PREFIX", "landing")
RAW_PREFIX = os.environ.get("RAW_PREFIX", "raw")
S3_BUCKET = os.environ.get("S3_BUCKET_NAME")
S3_REGION = os.environ.get("S3_BUCKET_REGION", "us-east-1")
FILE_TYPE = ".parquet"


class S3Uploader:
    def __init__(self, input_dir, bucket, prefix, file_format):
        self.input_dir = input_dir
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.file_format = file_format
        self.s3 = boto3.client("s3", region_name=S3_REGION)

    def upload_files(self):
        for root, dirs, files in os.walk(self.input_dir):
            for file in files:
                if file.endswith(self.file_format):
                    local_path = os.path.join(root, file)
                    relative_path = os.path.relpath(local_path, self.input_dir)
                    s3_path = f"{self.prefix}/{relative_path}".replace("\\", "/")

                    try:
                        self.s3.upload_file(local_path, self.bucket, s3_path)
                        print(f"Uploaded {local_path} to s3://{self.bucket}/{s3_path}")
                    except NoCredentialsError:
                        print("Credentials not available")
                        return


def main():
    local_raw = Path(DATA_DIR) / RAW_PREFIX
    uploader = S3Uploader(
        local_raw, bucket=S3_BUCKET, prefix=RAW_PREFIX, file_format=FILE_TYPE
    )
    uploader.upload_files()


if __name__ == "__main__":
    main()
