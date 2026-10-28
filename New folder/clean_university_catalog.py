import os
import re
import pandas as pd
import numpy as np


def extract_absolute_minimum_percentage(val, default=0.0):
    """
    Uses RegEx to pull numeric values from text. If a range or multiple numbers 
    are encountered, it strictly isolates and chooses the absolute minimum value.
    E.g., "55–65%" -> 55.0, "94-100" -> 94.0, "80% or 75% for State Boards" -> 75.0
    """
    if pd.isna(val):
        return default
    
    clean_val = str(val).replace(",", "").strip()
    
    # Find all standalone integers or decimals in the text line
    all_numbers = re.findall(r'(\d+(?:\.\d+)?)\s*%', clean_val)
    if not all_numbers:
        return default
        
    # Cast array items to floats and isolate the absolute minimum
    float_numbers = [float(n) for n in all_numbers if 0.0 < float(n) <= 100.0]
    if float_numbers:
        return min(float_numbers)
        
    return default


def parse_bachelor_percentage_regex(val):
    """
    Extracts the minimum bachelor marks threshold from the min_bachelor_requirement column.
    Safely parses percentages and structural NZ/Indian GPA ratios, prioritizing the minimum boundary.
    """
    if pd.isna(val):
        return 55.0  
    
    clean_val = str(val).strip().lower()
    
    # 1. Parse standard percentage expressions or ranges (choosing the minimum)
    pct_matches = re.findall(r"(\d+)\s*%", clean_val)
    if pct_matches:
        return min([float(p) for p in pct_matches])
        
    # 2. Parse structural GPA scales on a 9-point scale (GPE/GPA mapping)
    gpa_9_matches = re.findall(r"(?:gpa|gpe)\s*(\d+(?:\.\d+)?)\s*/\s*9", clean_val)
    if gpa_9_matches:
        min_gpa = min([float(g) for g in gpa_9_matches])
        return 50.0 + (min_gpa * 2.5)  # E.g., 4.0/9 -> 60.0%
        
    # 3. Parse structural GPA scales on a 4-point scale
    gpa_4_matches = re.findall(r"(?:gpa|gpe)\s*(\d+(?:\.\d+)?)\s*/\s*4", clean_val)
    if gpa_4_matches:
        min_gpa = min([float(g) for g in gpa_4_matches])
        return 50.0 + (min_gpa * 10.0)
        
    # 4. Semantic textual fallback mapping scales
    if "b+ average" in clean_val or "gpa 6.0+" in clean_val or "gpe 5.0" in clean_val:
        return 65.0
    if any(w in clean_val for w in ["b average", "second-class", "b- average", "first class"]):
        return 60.0
        
    # 5. Extract all numbers and locate the absolute minimum boundary float
    return extract_absolute_minimum_percentage(clean_val, default=55.0)


def dynamically_match_field_prereq(row):
    """
    Dynamically checks if the academic field mentioned in the 'field' column 
    is explicitly requested inside the 'academic_prerequisites' for that row.
    Handles semantic overlaps and splits multiple domain descriptors recursively.
    """
    if pd.isna(row["field"]) or pd.isna(row["academic_prerequisites"]):
        return 0
        
    prereq_text = str(row["academic_prerequisites"]).lower()
    field_text = str(row["field"]).lower()
    
    # Clean and split compound domain headers (e.g., "Computer Science / IT" -> ["computer science", "it"])
    fields_to_check = [f.strip() for f in re.split(r'[/|&,]', field_text) if f.strip() != ""]
    
    # Broad classification catch for generalist/uncategorised streams
    if any(unendorsed in fields_to_check for unendorsed in ["other", "short course / other", "arts / humanities"]):
        return 0
        
    # Check if any structural token substring matches the baseline prerequisites block
    for item in fields_to_check:
        if len(item) > 2 and item in prereq_text:
            return 1
        if item in ["it", "ai", "cs"] and re.search(r'\b' + re.escape(item) + r'\b', prereq_text):
            return 1
            
    return 0


def parse_intl_eligibility(val):
    if pd.isna(val):
        return 0
    clean_val = str(val).strip().upper()
    if "NOT ELIGIBLE" in clean_val or "DOMESTIC ONLY" in clean_val:
        return 0
    if "ELIGIBLE" in clean_val or "OPEN" in clean_val:
        return 1
    return 0


def parse_work_experience(req_text):
    if pd.isna(req_text):
        return 0, 0
    clean_text = str(req_text).strip().lower()
    if clean_text in ["none", "no", "none specified"] or "not formally mandated" in clean_text:
        return 0, 0
    
    is_mandatory = 1 if any(w in clean_text for w in ["required", "must be", "mandatory", "yes"]) else 0
    
    year_match = re.search(r"(\d+)\+?\s*year", clean_text)
    if year_match:
        return is_mandatory, int(year_match.group(1)) * 12
        
    return is_mandatory, 0


def extract_bachelor_years_regex(combined_context):
    """
    Deterministic rule-based RegEx scanner replacing the LLM.
    Scans the unified text block for strict flags indicating a 4-year degree is mandatory.
    """
    clean_text = str(combined_context).lower()
    
    # Rules targeting strict 4-year constraints
    four_year_indicators = [
        r"four-year", 
        r"4-year", 
        r"be/btech", 
        r"b\.tech", 
        r"b\.e\.",
        r"honours degree", 
        r"barch", 
        r"b\.arch",
        r"insufficient and would need topping up",
        r"three-year.*insufficient"
    ]
    
    # Exceptions that explicitly state an Indian 3-year degree is okay
    three_year_allowed_indicators = [
        r"3-year.*accepted", 
        r"3-year or 4-year.*accepted", 
        r"3-year.*is fine", 
        r"open to non-accounting graduates",
        r"conversion master"
    ]
    
    # Evaluate 3-year allowance overrides first
    for pattern in three_year_allowed_indicators:
        if re.search(pattern, clean_text):
            return 3
            
    # Evaluate strict 4-year boundaries
    for pattern in four_year_indicators:
        if re.search(pattern, clean_text):
            return 4
            
    return 3  # Standard system fallback


def extract_work_exp_substitution_regex(combined_context):
    """
    Deterministic rule-based RegEx scanner replacing the LLM.
    Identifies if alternative work experience context allows a degree requirement waiver.
    """
    clean_text = str(combined_context).lower()
    
    substitution_patterns = [
        r"experience may substitute",
        r"experience can substitute",
        r"experience in lieu",
        r"professional practice may substitute",
        r"experience.*accepted as an alternative entry",
        r"experience can substitute for standard grades",
        r"experience.*substitute for missing subject"
    ]
    
    for pattern in substitution_patterns:
        if re.search(pattern, clean_text):
            return 1
            
    return 0


def execute_university_catalog_cleaning(catalog_path, output_path):
    """
    Orchestration routine executing the entire deterministic university data normalization pipeline.
    """
    print(f"[INFO] Initializing rule-based extraction framework for path: {catalog_path}...")
    if not os.path.exists(catalog_path):
        raise FileNotFoundError(f"Source catalog dataframe missing at path: {catalog_path}")
        
    df = pd.read_csv(catalog_path)
    
    # 1. Deterministic RegEx Transformations (Isolating Absolute Minimums)
    df["is_intl_eligible"] = df["programme_eligibility_intl"].apply(parse_intl_eligibility)
    df["min_std12_pct"] = df["min_std12_%"].apply(lambda x: extract_absolute_minimum_percentage(x, default=0.0))
    df["min_bachelor_pct"] = df["min_bachelor_requirement"].apply(parse_bachelor_percentage_regex)
    
    df["ielts_min_overall"] = df["ielts_overall"].apply(lambda x: extract_absolute_minimum_percentage(x, default=6.0))
    df["ielts_min_band"] = df["ielts_min_band"].apply(lambda x: extract_absolute_minimum_percentage(x, default=5.5))
    df["pte_min_overall"] = df["pte"].apply(lambda x: extract_absolute_minimum_percentage(x, default=50.0))
    df["toefl_min_overall"] = df["toefl"].apply(lambda x: extract_absolute_minimum_percentage(x, default=80.0))
    
    work_params = df["work_experience_required"].apply(parse_work_experience)
    df["work_exp_mandatory"] = [p[0] for p in work_params]
    df["min_work_exp_months"] = [p[1] for p in work_params]
    
    # 2. Execute Dynamic Field-Prerequisites Check Block
    df["field_prereq_match"] = df.apply(dynamically_match_field_prereq, axis=1)
    
    # 3. Compile Multi-Column Composite Strings for Structural Rule Parsing
    academic_str = df["academic_prerequisites"].fillna("").astype(str)
    entrance_str = df["entrance_requirements"].fillna("").astype(str)
    eligibility_str = df["eligibility_note"].fillna("").astype(str)
    
    # Build text block mapping values to clean out redundant duplicates
    combined_context_series = academic_str + " | " + entrance_str + " | " + eligibility_str
    
    # Execute deterministic matching across the aggregated multi-column context array
    df["min_bachelor_yrs"] = combined_context_series.apply(extract_bachelor_years_regex).astype(int)
    df["substitute_bachelor_with_work_exp"] = combined_context_series.apply(extract_work_exp_substitution_regex).astype(int)
    
    df["programme_clean"] = df["programme"].astype(str).str.strip()
    df["university_clean"] = df["university"].astype(str).str.strip()
    
    # Save the output to disk
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"[SUCCESS] Pure rule-based university catalog successfully outputted to: {output_path}")
    print(f"Total available offerings processed: {len(df)}")


if __name__ == "__main__":
    RAW_CATALOG = "data/raw/University_database.csv"
    PROCESSED_CATALOG = "data/processed/university_catalog_clean.csv"
    execute_university_catalog_cleaning(RAW_CATALOG, PROCESSED_CATALOG)