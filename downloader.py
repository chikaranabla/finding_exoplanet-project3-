import kagglehub
import shutil
import os


path = kagglehub.dataset_download("vijayveersingh/kepler-and-tess-exoplanet-data")


csv_file = os.path.join(path, "q1_q8_koi_2025.02.03_04.12.15.csv")
shutil.copy(csv_file, "./q1_q8_koi_2025.02.03_04.12.15.csv")

print(f"Copied file to: {csv_file}")