import re
import io
import pandas as pd

def read_avl_csv(path: str):
    """
    Parse an AVL-style CSV with [HEADER START]/[DATA START] blocks.
    Returns a DataFrame with columns: RecordingTime (s), DateTime, PEC_Measured_Voltage,
    PEC_Measured_Current, ... plus cell voltages if present.
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    m = re.search(r"\[DATA START\](.*)", text, flags=re.S)
    if not m:
        raise ValueError("Could not find [DATA START] section in file.")
    
    csv_part = m.group(1).strip()
    # Some files include unit and dtype lines; pandas handles them as header rows or we skip them:
    # We will read fully and drop rows where first column is non-numeric.
    df = pd.read_csv(io.StringIO(csv_part), header=None)
    # Drop non-numeric "header rows" in the data part

    def is_numeric(x):
        try:
            float(x)
            return True
        except:
            return False

    # Keep rows where first col is numeric index
    mask = df[0].astype(str).apply(is_numeric)
    df = df[mask].reset_index(drop=True)

    # Re-read with the known column names from your specimen (trimmed)
    # Adjust columns to your file; here we use the header slice shown earlier.
    cols = [
        "RecordingTime","DateTime","PEC_Measured_Voltage","PEC_Measured_Current",
        "PEC_Measured_Power","Timer","Timer_2","Count","Count_2","Vmax","Vmin",
        "HVBatCellVlt_12","HVBatCellVlt_11","HVBatCellVlt_10","HVBatCellVlt_09",
        "HVBatCellVlt_08","HVBatCellVlt_07","HVBatCellVlt_06","HVBatCellVlt_05",
        "HVBatCellVlt_04","HVBatCellVlt_03","HVBatCellVlt_02","HVBatCellVlt_01"
    ]

    df.columns = cols[:df.shape[1]]
    # Convert types
    df["RecordingTime"] = pd.to_numeric(df["RecordingTime"], errors="coerce")
    df["PEC_Measured_Voltage"] = pd.to_numeric(df["PEC_Measured_Voltage"], errors="coerce")
    df["PEC_Measured_Current"] = pd.to_numeric(df["PEC_Measured_Current"], errors="coerce")

    return df

def dataframe_from_input(path: str):
    if path.lower().endswith(".csv"):
        # assume AVL CSV
        df = read_avl_csv(path)
        # unify column names
        out = df.rename(columns={
            "RecordingTime":"t",
            "PEC_Measured_Voltage":"V",
            "PEC_Measured_Current":"I"
        })

        out["T"] = 25.0  # if temperature not present; user can merge later
        return out[["t","V","I","T"]].dropna()
    elif path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")