import os
import re
import numpy as np
import pandas as pd


def standardize_percentage(val):
    """
    Standardizes decimal markers (e.g., 0.872 -> 87.2) into a standard 0-100% scale.
    Handles string parsing anomalies and filters out systemic invalid inputs.
    """
    if pd.isna(val):
        return np.nan
    try:
        # Cast value to float
        num = float(val)
        # If expressed as a decimal coefficient < 1.0 (e.g. 0.941), upscale to percent
        if 0.0 < num <= 1.0:
            return round(num * 100, 2)
        elif num > 1.0 <= 100.0:
            return round(num, 2)
        return np.nan
    except (ValueError, TypeError):
        return np.nan


def parse_backlogs(val):
    """
    Defensive parser to protect backlog fields against floating-point bleed.
    Ensures text decimals are handled safely without inflating values.
    """
    if pd.isna(val):
        return 0

    # Convert to string and strip whitespace
    val_str = str(val).strip()

    # CRITICAL: If a float leaked in, drop the fractional decimals completely first!
    if "." in val_str:
        val_str = val_str.split(".")[0]

    # Clean any remaining textual debris
    cleaned = re.sub(r"[^\d]", "", val_str)

    if cleaned == "":
        return 0

    return int(cleaned)


def clean_english_tests(df):
    """
    Normalizes test type categorizations and separates raw bands from totals.
    Implements a strict categorical enumeration framework.
    """
    # Standardize textual naming flags
    df["Test type"] = df["Test type"].astype(str).str.strip().str.upper()
    df["Test type"] = df["Test type"].replace(
        {
            "IELTS ACADEMIC": "IELTS",
            "PTE ACADEMIC": "PTE",
            "TOEFL IBT": "TOEFL",
            "OTHERS": "OTHERS",
            "NAN": "NONE",
            "NOT ASSIGNED": "NONE",
        }
    )

    # Coerce total score metrics into strict continuous float parameters
    df["Total Score"] = pd.to_numeric(df["Total Score"], errors="coerce").fillna(0.0)
    
    return df


def execute_academic_cleaning_pipeline(student_df_path, output_df_path):
    """
    Orchestrates the transactional execution loop for data cleaning.
    """
    print(f"[INFO] Ingesting source records from target: {student_df_path}...")
    
    if not os.path.exists(student_df_path):
        raise FileNotFoundError(f"Source student log not discovered at path: {student_df_path}")
        
    #df = pd.read_csv(student_df_path)
    df = pd.read_excel(student_df_path, engine="openpyxl", sheet_name=0)

    # 1. Clean and standardize continuous academic percentile fields
    df["12th marks_clean"] = df["12th marks"].apply(standardize_percentage)
    df["bachelors marks_clean"] = df["bachelors marks"].apply(standardize_percentage)

    # 2. Parse out structural backlog vectors
    df["number of backlogs_clean"] = df["number of backlogs"].apply(parse_backlogs)

    # 3. Process categorical English standard metrics
    df = clean_english_tests(df)

    # Drop missing metrics or rows that lack baseline qualification targets
    # For training data accuracy, we drop profiles missing critical academic anchors
    #df = df.dropna(subset=["12th marks_clean", "bachelors marks_clean"], how="any")

    # Ensure targeted output directory layout exists seamlessly
    os.makedirs(os.path.dirname(output_df_path), exist_ok=True)
    
    # Commit changes out to tracking directory storage
    df.to_csv(output_df_path, index=False)
    print(f"[SUCCESS] Exported cleaned historical user table to: {output_df_path}")
    print(f" -> Retained profiles for system training: {len(df)} rows.")


if __name__ == "__main__":
    # Test path endpoints running inside root directory layout configurations
    RAW_PATH = "data/raw/Student_Data_NZ_analysis.xlsx"
    PROCESSED_PATH = "data/processed/students_clean.csv"
    
    # Fallback to test execution engine local runtime metrics verification loop
    try:
        execute_academic_cleaning_pipeline(RAW_PATH, PROCESSED_PATH)
    except Exception as e:
        print(f"[ERROR] Engine cleaning loop aborted context execution: {str(e)}")
