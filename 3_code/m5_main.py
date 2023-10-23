import pandas as pd
from pathlib import Path

home_dir = Path(__file__).parent
input_folder = home_dir / 'input'
sample_submission = pd.read_csv(input_folder / 'sample_submission.csv')
sample_submission.head(10)