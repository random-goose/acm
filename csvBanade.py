import pandas as pd

def tsv_to_csv(tsv_file, csv_file):
    df = pd.read_csv(tsv_file, delimiter='\t')
    df.to_csv(csv_file, index=False)

# Example usage
tsv_to_csv('diabetes.tsv', 'diabetes.csv')
print('hogya')