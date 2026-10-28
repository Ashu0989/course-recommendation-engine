import os
import re
import json
import pandas as pd
from thefuzz import fuzz, process


class TextNormalizerEntityResolver:
    """
    Programmatic text pre-processing and semantic string alignment engine.
    Standardizes course descriptions and executes fuzzy entity resolution 
    to map raw application tokens cleanly back onto standard catalog IDs.
    """

    def __init__(self, config_path="config/settings.yaml"):
        self.min_fuzzy_score = 85  # Standard configuration fallback threshold
        self.mapping_output_path = "data/mappings/course_name_mapping.json"
        
        # Safe structural initialization if running outside orchestration loops
        if os.path.exists(config_path):
            try:
                import yaml
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
                    self.min_fuzzy_score = config.get("thresholds", {}).get("fuzzy_match_min_score", 85)
                    self.mapping_output_path = config.get("data_paths", {}).get("course_mapping", self.mapping_output_path)
            except Exception as e:
                print(f"[WARNING] Settings file read failed, using system defaults: {str(e)}")

    @staticmethod
    def preprocess_string(text):
        """
        Pure programmatic text normalization engine.
        Applies downcasing, strips structural noise characters, and discards
        systemic programmatic filler terms to reveal core text strings.
        """
        if pd.isna(text):
            return ""
        
        # Enforce unified casing and strip trailing spatial boundaries
        clean = str(text).lower().strip()
        
        # Remove parentheses debris and trailing qualification details
        clean = re.sub(r"\(.*?\)", "", clean)
        
        # Remove common operational tracking tokens and stop-words
        generic_fillers = [
            r"\btaught\b", r"\bpoints\b", r"\bpoint\b", r"\blevel\b", 
            r"\bcredits\b", r"\bcredit\b", r"\bresearch\b", r"\bsector\b",
            r"\b pathway\b", r"\b pathway -\b", r"-\b taught\b"
        ]
        for pattern in generic_fillers:
            clean = re.sub(pattern, "", clean)
            
        # Clean out non-alphanumeric punctuation boundaries
        clean = re.sub(r"[^\w\s-]", "", clean)
        
        # Condense multiple spatial blocks down to a single formatting split
        clean = re.sub(r"\s+", " ", clean).strip()
        
        return clean

    def generate_fuzzy_entity_mapping(self, clean_students_path, clean_catalog_path):
        """
        Calculates string distances across unaligned course vectors.
        Resolves semantic name discrepancies and dumps an explicit mapping matrix file.
        """
        print("[INFO] Initializing Fuzzy Entity Resolution Pipeline...")
        
        if not os.path.exists(clean_students_path) or not os.path.exists(clean_catalog_path):
            raise FileNotFoundError("[ERROR] Input cleaned data dependencies missing for name reconciliation.")

        students_df = pd.read_csv(clean_students_path)
        catalog_df = pd.read_csv(clean_catalog_path)

        # Isolate distinct strings tracking from the data layers
        raw_app_courses = students_df["Opportunity Course Name"].dropna().unique().tolist()
        
        # Zip clean catalog titles along with their unique row index numbers
        catalog_pool = catalog_df[["id", "programme"]].dropna().to_dict(orient="records")
        catalog_mapping_dict = {self.preprocess_string(item["programme"]): (item["id"], item["programme"]) for item in catalog_pool}
        catalog_clean_names = list(catalog_mapping_dict.keys())

        resolved_lookup_map = {}
        unresolved_anomalies_count = 0

        print(f"[INFO] Computing Levenshtein matrices over {len(raw_app_courses)} candidate names...")

        for raw_name in raw_app_courses:
            clean_app_name = self.preprocess_string(raw_name)
            
            if not clean_app_name:
                continue

            # Run a rapid token ratio fuzzy extraction sequence over our catalog name matrix
            best_match = process.extractOne(clean_app_name, catalog_clean_names, scorer=fuzz.token_set_ratio)
            
            if best_match and best_match[1] >= self.min_fuzzy_score:
                matched_clean_key = best_match[0]
                catalog_id, standard_name = catalog_mapping_dict[matched_clean_key]
                
                # Commit valid structural link profile to map dictionary
                resolved_lookup_map[raw_name] = {
                    "catalog_id": int(catalog_id),
                    "standardized_name": standard_name,
                    "confidence_score": int(best_match[1]),
                    "resolution_method": "deterministic_fuzzy_regex"
                }
            else:
                unresolved_anomalies_count += 1
                # Mark lower assurance entries clearly to allow safe manual overriding additions
                resolved_lookup_map[raw_name] = {
                    "catalog_id": None,
                    "standardized_name": None,
                    "confidence_score": int(best_match[1]) if best_match else 0,
                    "resolution_method": "manual_review_required"
                }

        # Save structured dictionary configurations back to file storage
        os.makedirs(os.path.dirname(self.mapping_output_path), exist_ok=True)
        with open(self.mapping_output_path, "w", encoding="utf-8") as f:
            json.dump(resolved_lookup_map, f, indent=4, ensure_ascii=False)

        print(f"[SUCCESS] Mapping file updated: {self.mapping_output_path}")
        print(f" -> Automatically Matched Tracks: {len(raw_app_courses) - unresolved_anomalies_count}")
        print(f" -> Flagged Ambiguous Anomalies: {unresolved_anomalies_count} (Saved with catalog_id: null for safety)")
        
        return resolved_lookup_map


if __name__ == "__main__":
    # Test path pointers checking operational alignment parameters
    STUDENTS_CLEAN_FILE = "data/processed/students_clean.csv"
    CATALOG_CLEAN_FILE = "data/processed/university_catalog_clean.csv"
    
    normalizer = TextNormalizerEntityResolver()
    
    try:
        normalizer.generate_fuzzy_entity_mapping(STUDENTS_CLEAN_FILE, CATALOG_CLEAN_FILE)
    except Exception as e:
        print(f"[ERROR] Text normalization routine aborted execution context: {str(e)}")
