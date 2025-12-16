import numpy as np
import os
import pandas as pd

num_pos_total = 0
num_neg_total = 0

#for i in range(1, 3):
#    npz_path = f"kepler_dataset_v1/batch_{i}.npz"

#    data = np.load(npz_path)
#    print(data.files)
#    flux_global = data['flux_global']
#    flux_local = data['flux_local']
#    label = data['label']


DATA_DIR = "kepler_dataset_v1"
meta = pd.read_csv(os.path.join(DATA_DIR, "final_metadata.csv"))

for i in range(len(meta)):
    r = meta.iloc[i]
    npz = np.load(os.path.join(DATA_DIR, r["filename"]))
    k = int(r["index_in_batch"])
    meta_label = r["label"]
    npz_label = float(npz["label"][k])
    if meta_label != npz_label:
        print("meta label and npz label are not the same")
    
    
print(f"done checking {len(meta)} rows")

# Count labels (0 and 1) for batch_1.npz and batch_2.npz
for batch_id in (1, 2):
    npz_path = os.path.join(DATA_DIR, f"batch_{batch_id}.npz")
    data = np.load(npz_path)
    labels = data["label"]
    num_zero = int(np.sum(labels == 0))
    num_one = int(np.sum(labels == 1))
    print(f"batch_{batch_id}.npz -> label 0: {num_zero}, label 1: {num_one}")
