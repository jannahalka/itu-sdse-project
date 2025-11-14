import os
import subprocess

from itu_sdse_project.config import RAW_DATA_DIR

FILE_NAME =  "raw_data.csv"

if not os.path.exists(RAW_DATA_DIR / FILE_NAME):
    subprocess.run(["dvc", "get", "https://github.com/Jeppe-T-K/itu-sdse-project-data", FILE_NAME, "-o", RAW_DATA_DIR])
    print("succesfully downloaded")
else:
    print("file already exists")
