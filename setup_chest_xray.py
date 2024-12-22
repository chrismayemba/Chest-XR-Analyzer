import os

# Base project directory
base_dir = "c:/Users/chris/CascadeProjects/chest_xray_analyzer"

# Create directory structure
directories = [
    "",  # Base directory
    "data",
    "data/raw",
    "data/processed",
    "models",
    "src",
    "src/preprocessing",
    "src/models",
    "src/visualization",
    "src/utils",
    "notebooks",
    "config",
    "tests",
    "reports",
    "docs"
]

for dir_path in directories:
    full_path = os.path.join(base_dir, dir_path)
    os.makedirs(full_path, exist_ok=True)
    print(f"Created directory: {full_path}")
