import pandas as pd

# Input/output files
input_file = "C:/Users/jwiegner/OneDrive - ETH Zurich/00_Papers_Journal/00_2026-Quadratic terms in ESM/IEA TRL list/IEA_Clean_Tech_Guide.csv"
output_file = "C:/Users/jwiegner/OneDrive - ETH Zurich/00_Papers_Journal/00_2026-Quadratic terms in ESM/IEA TRL list/IEA_Clean_Tech_Guide_formatted.csv"

n = 12
# Read entire file as one long string
raw_df = pd.read_csv(input_file, header=None)
arr = raw_df.iloc[0].to_numpy()
chunks = [arr[i:i+n] for i in range(0, len(arr), n)]

reshaped = pd.DataFrame(chunks)

# Save nicely formatted CSV
reshaped.to_csv(output_file, index=False)
