import os
import subprocess

from itu_sdse_project.config import RAW_DATA_DIR
from loguru import logger

FILE_NAME =  "raw_data.csv"

if __name__ == "__main__":
    target_path = RAW_DATA_DIR / FILE_NAME

    if not os.path.exists(target_path):
        logger.info("Raw data not found at {}. Downloading with dvc...", target_path)

        result = subprocess.run(["dvc", "get", "https://github.com/Jeppe-T-K/itu-sdse-project-data", FILE_NAME, "-o", RAW_DATA_DIR], check=False)

        if result.returncode == 0:
            logger.success("Successfully downloaded raw data to {}", target_path)
        else:
            logger.error("Failed to download raw data (exit code {}).", result.returncode)
    else:
        logger.info("Raw data already exists at {}. Skipping download.", target_path)
