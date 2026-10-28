import json
from pathlib import Path

import re
import numpy as np
import pandas as pd

data_file = Path(__file__).parent / "data" / "University_database.xlsx"
univ_df = pd.read_excel(data_file, sheet_name=0)
# print("DataFrame headers:")
# print(univ_df.columns.tolist())


# 1. Extract minimum 12th percentage
def extract_min_percentage(text):
  if pd.isna(text):
    return np.nan
  matches = re.findall(r'(\d+(?:\.\d+)?)\s*%', str(text))
  if matches:
    return min([float(m) for m in matches])
  return np.nan

# 2. Extract bachelor requirement percentage / CGPA
def extract_bachelor_min_percentage(text):
  if pd.isna(text):
    return np.nan
  text_str = str(text)
  pct_matches = re.findall(r'(\d+(?:\.\d+)?)\s*%', text_str)
  if pct_matches:
    return float(pct_matches[0])
  cgpa_matches = re.findall(r'CGPA\s*(\d+(?:\.\d+)?)', text_str, re.IGNORECASE)
  if cgpa_matches:
    return float(cgpa_matches[0]) * 10  # Approximate CGPA 6.0 -> 60%
  return np.nan

def extract_bachelor_substitutes(text):
  if pd.isna(text):
    return 0
  text_str = str(text).lower()
  if ('professional' in text_str or 'experience' in text_str or 'substitute' in text_str or 'equivalent' in text_str):
    return 1
  return 0

# 3. Extract work experience years
def extract_work_exp(text):
  if pd.isna(text):
    return 0.0
  text_str = str(text).lower()
  if 'none' in text_str or 'no ' in text_str or text_str.strip() == 'no':
    return 0.0
  matches = re.findall(r'(\d+)\+?\s*(?:year|yr)', text_str)
  if matches:
    return float(matches[0])
  return 0.0

# 4. Extract cognate degree requirement flag
def extract_cognate_requirement(text):
  if pd.isna(text):
    return 0
  text_str = str(text).lower()
  if 'cognate' in text_str or 'relevant' in text_str or 'same subject' in text_str:
    return 1
  return 0


# 5. Extract same field requirement flag
def extract_same_field_requirement(text1, text2):
  if pd.isna(text1) or pd.isna(text2):
    return 0
  text1_str = str(text1).lower()
  text2_str = str(text2).lower()
  if text1_str in text2_str or 'same subject' in text1_str:
    return 1
  return 0

# Apply extractions to DataFrame
univ_df['clean_min_12th'] = univ_df['min_std12_%'].apply(extract_min_percentage)
univ_df['clean_min_bachelor'] = univ_df['min_bachelor_requirement'].apply(extract_bachelor_min_percentage)


univ_df['clean_work_exp'] = univ_df['work_experience_required'].apply(extract_work_exp)

univ_df['clean_ielts'] = pd.to_numeric(univ_df['ielts_overall'], errors='coerce')
univ_df['clean_pte'] = pd.to_numeric(univ_df['pte'], errors='coerce')
univ_df['clean_toefl'] = pd.to_numeric(univ_df['toefl'], errors='coerce')

univ_df['cognate_degree_required'] = (
    univ_df['academic_prerequisites'].apply(extract_cognate_requirement)
    | univ_df['entrance_requirements'].apply(extract_cognate_requirement)
)

univ_df['same_field_required'] = univ_df.apply(
    lambda row: extract_same_field_requirement(row['field'], row['academic_prerequisites']),
    axis=1
)

univ_df['clean_bachelor_substitutes'] = univ_df['academic_prerequisites'].apply(extract_bachelor_substitutes)
print(
    'Phase 1 Cleaning Complete. Master Eligibility Table Shape:'
    f' {univ_df.shape}'
)

print("Saving cleaned data to 'cleaned_university_data.csv'...")
# Save the cleaned DataFrame to a CSV file
univ_df.to_csv('cleaned_university_data.csv', index=False)
