import pandas as pd
import glob
import os

# Check batch files
files = sorted(glob.glob('pipeline_output_v5/batch_*_meta.csv'))
print(f'Found {len(files)} batch files\n')

total_label1 = 0
total_label0 = 0

for f in files:
    try:
        df = pd.read_csv(f)
        label1 = len(df[df['label'] == 1])
        label0 = len(df[df['label'] == 0])
        total_label1 += label1
        total_label0 += label0
        print(f'{os.path.basename(f)}: Total={len(df)}, Label1={label1}, Label0={label0}')
    except Exception as e:
        print(f'Error reading {f}: {e}')

print(f'\nTotal: Label1={total_label1}, Label0={total_label0}')

# Check final metadata
if os.path.exists('pipeline_output_v5/final_metadata.csv'):
    df_final = pd.read_csv('pipeline_output_v5/final_metadata.csv')
    print(f'\nFinal metadata: Total={len(df_final)}, Label1={len(df_final[df_final["label"]==1])}, Label0={len(df_final[df_final["label"]==0])}')



