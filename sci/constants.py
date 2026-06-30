import os

RDS_PATH_COL = "Path"
RDS_TASKS = ["rds", "No Finding", "Finding"]

CHEXPERT_PATH_COL = "Path"
CHEXPERT_COMPETITION_TASKS = ["No Finding", "Finding"]
CHEXPERT_UNCERTAIN_MAPPINGS = { -1: 0 }
CHEXPERT_DATA_DIR = os.environ.get("CHEXPERT_DATA_DIR", "Data")
