import pandas as pd
from tqdm import tqdm
from radio2mmsi import resolve_radio
import argparse
import os

def get_callsigns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keeps only unique radio callsigns in the dataframe
    """
    df = df.loc[:, ["Radiokallesignal (ERS)"]]
    df = df.dropna()
    df = df.drop_duplicates()
    return df

def radio2mmsi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resolve the MMSI of a vessel based on the radio callsign
    """
    df["mmsi"] = 0
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Resolving MMSIs"):
        df.loc[idx, "mmsi"] = resolve_radio(row["Radiokallesignal (ERS)"])
    df = df.dropna()
    df["mmsi"] = df["mmsi"].astype(int)
    return df

def main(por_file: str) -> None:
    """
    Create mapping between the radio callsign and MMSI of a vessel
    """
    output_file = "radio2mmsi.csv"
    
    # Load new data
    por_df = pd.read_csv(por_file, sep=";", decimal=",", low_memory=False)
    por_radio = get_callsigns(por_df)
    new_df = radio2mmsi(por_radio)
    
    # Load existing data if file exists
    if os.path.exists(output_file):
        print(f"Loading existing mappings from {output_file}")
        existing_df = pd.read_csv(output_file, sep=";")
        
        # Combine old and new data
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        
        # Remove duplicates, keeping the first occurrence
        combined_df = combined_df.drop_duplicates(subset=["Radiokallesignal (ERS)"], keep="first")
        
        print(f"Added {len(new_df)} new entries, total: {len(combined_df)} unique callsigns")
    else:
        print(f"Creating new file {output_file}")
        combined_df = new_df
    
    # Save combined data
    combined_df.to_csv(output_file, index=False, sep=";")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a mapping from radio callsign to MMSI")
    parser.add_argument("por", type=str, help="Path to the POR dataset")
    args = parser.parse_args()
    main(args.por)