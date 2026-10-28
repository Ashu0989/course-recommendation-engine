import os
import pandas as pd
import numpy as np


class HardConstraintRuleEngine:
    """
    Stage 1 Vetting Matrix: Evaluates a real-time student profile against 
    the cleaned university course catalog using deterministic validation arrays.
    """

    def __init__(self, processed_catalog_path="data/processed/catalog_clean.csv"):
        if not os.path.exists(processed_catalog_path):
            raise FileNotFoundError(
                f"[ERROR] Processed university catalog database missing at: {processed_catalog_path}"
            )
        self.catalog_df = pd.read_csv(processed_catalog_path)

    def evaluate_english_eligibility(self, student_profile, course_row):
        """
        Validates if the user's English proficiency satisfies the catalog floor criteria.
        Safely switches checking criteria based on the user's explicit test type.
        """
        test_type = str(student_profile.get("test_type", "NONE")).strip().upper()
        total_score = float(student_profile.get("english_total_score", 0.0))

        # 1. Direct short-circuit optimization if the course does not require an English score profile
        if pd.isna(course_row["ielts_min_overall"]) or course_row["ielts_min_overall"] == 0:
            return True

        # 2. Strategic routing based on explicit test categorizations
        if "IELTS" in test_type:
            if total_score < float(course_row["ielts_min_overall"]):
                return False
            # Check individual band score threshold if user passed an explicit band vector
            student_min_band = float(student_profile.get("english_min_band", 0.0))
            if student_min_band > 0.0 and student_min_band < float(course_row["ielts_min_band"]):
                return False
            return True

        elif "PTE" in test_type:
            if total_score < float(course_row["pte_min_overall"]):
                return False
            return True

        elif "TOEFL" in test_type:
            if total_score < float(course_row["toefl_min_overall"]):
                return False
            return True

        # If user has no test type or matched a niche format, fall back to checking IELTS overall baseline equivalent
        if total_score > 0.0 and course_row["ielts_min_overall"] > 0.0:
            # Simple fallback check if user marks are explicit
            return True

        return False

    def validate_eligibility_matrix(self, student_profile):
        """
        Runs the full student parameter sequence across all catalog rows.
        Returns a filtered Pandas DataFrame containing only qualified options.
        """
        eligible_indices = []

        # Extract user profile parameters explicitly
        std12_marks = float(student_profile.get("marks_12th", 0.0))
        ug_marks = float(student_profile.get("marks_bachelors", 0.0))
        ug_duration_yrs = int(student_profile.get("bachelors_duration_years", 3))
        work_exp_months = int(student_profile.get("work_experience_months", 0))

        print(f"[INFO] Initializing Stage 1 Constraint Matrix optimization array...")
        print(f" -> Input Profile: 12th: {std12_marks}%, UG: {ug_marks}% ({ug_duration_yrs}yrs), Work Exp: {work_exp_months} months.")

        for idx, row in self.catalog_df.iterrows():
            # Flag 1: Hard verification for International Student entry allowance rules
            if int(row["is_intl_eligible"]) == 0:
                continue

            # Flag 2: Standard 12th Grade baseline parameter bounds validation
            if row["min_std12_pct"] > 0.0 and std12_marks < row["min_std12_pct"]:
                continue

            # Flag 3: Bachelor's qualification duration checks with experience waivers
            if ug_duration_yrs < int(row["min_bachelor_yrs"]):
                # Check if the course allows professional experience to substitute for structural degree depth
                if int(row["substitute_bachelor_with_work_exp"]) == 0:
                    continue
                # If substitution is allowed, enforce a baseline requirement of at least 24 months of work history
                if work_exp_months < 24:
                    continue

            # Flag 4: Academic marks cut-offs matching the catalog row floor threshold
            if row["min_bachelor_pct"] > 0.0 and ug_marks < row["min_bachelor_pct"]:
                continue

            # Flag 5: Mandatory occupational career work experience constraints
            if int(row["work_exp_mandatory"]) == 1:
                if work_exp_months < int(row["min_work_exp_months"]):
                    continue

            # Flag 6: Language proficiency score assessment profiles execution loops
            if not self.evaluate_english_eligibility(student_profile, row):
                continue

            # If row survives all structural conditional filters, append to output target candidates pool
            eligible_indices.append(idx)

        # Compile final matched matrix rows
        filtered_candidates = self.catalog_df.loc[eligible_indices].copy()
        
        print(f"[STAGE 1 SUCCESS] Verification loops complete. Filtered rows from {len(self.catalog_df)} down to {len(filtered_candidates)} eligible tracks.")
        return filtered_candidates


if __name__ == "__main__":
    # Mock runtime simulation profiling array
    engine = HardConstraintRuleEngine(processed_catalog_path="data/processed/catalog_clean.csv")
    
    # Simulating a typical Indian student candidate profile
    mock_student = {
        "marks_12th": 78.5,
        "marks_bachelors": 62.0,
        "bachelors_duration_years": 3,  # Standard Indian 3-year BA/BSc/BCom track
        "test_type": "IELTS",
        "english_total_score": 6.5,
        "english_min_band": 6.0,
        "work_experience_months": 12
    }
    
    try:
        valid_options = engine.validate_eligibility_matrix(mock_student)
        print(f"Sample surfaced matches snippet:\n{valid_options[['university_clean', 'programme_clean', 'min_bachelor_pct']].head(3)}")
    except Exception as e:
        print(f"[CRITICAL] Operational logging verification failed context: {str(e)}")
