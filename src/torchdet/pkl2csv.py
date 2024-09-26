import pandas as pd
import pickle
from pathlib import Path
import sys

def pkl_to_csv(pkl_path, csv_path=None):
    try:
        data = pd.DataFrame(pickle.load(open(pkl_path, 'rb')))
        if csv_path is None:
            csv_path = pkl_path.with_suffix('.csv')
        data.to_csv(csv_path, index=False)
        print(f"Successfully saved to {csv_path}")
    except (FileNotFoundError, pickle.UnpicklingError, pd.errors.EmptyDataError) as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python pkl_to_csv.py <input-pkl-file> [output-csv-file]")
    
    pkl_path = Path(sys.argv[1])
    csv_path = Path(sys.argv[2]) if len(sys.argv) == 3 else None
    pkl_to_csv(pkl_path, csv_path)
