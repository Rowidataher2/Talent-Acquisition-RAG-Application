import os
import re

job_titles = [
    "SW_MLEngineer", "SW_ML", "DataScientist", "DataEngineer", "DataTesting",
    "BusinessIntelligenceDeveloper", "Resume", "CV"
]

def extract_name(filename):
    # Remove directory path and file extension
    name_part = os.path.splitext(os.path.basename(filename))[0]

    # Remove job titles
    for title in job_titles:
        name_part = name_part.replace(title, "").strip("_").strip()

    # Replace underscores with spaces
    name_part = name_part.replace("_", " ")

    # Remove any special characters except spaces
    name_part = re.sub(r'[^A-Za-z ]', '', name_part)

    # Add space before capital letters if not preceded by a space
    name_part = re.sub(r'(?<=[a-z])([A-Z])', r' \1', name_part)

    # Remove extra spaces
    name_part = re.sub(r'\s+', ' ', name_part).strip()

    return name_part