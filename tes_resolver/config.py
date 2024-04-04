import datetime
from pathlib import Path

time_stamp = datetime.datetime.now().strftime("%Y-%m-%d(%H-%M-%S.%f)")

home_dir = Path(__file__).resolve().parent.__str__()

