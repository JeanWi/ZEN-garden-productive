import pandas as pd
from pathlib import Path
import os
from preprocessing.helpers import ModelApi, generate_delta_xr_technologies


root_path = Path("C:/ZenGardenInput/ZEN-models/data")

nr_timesteps = 720
dataset = "Crystal_Ball"
os.chdir(root_path)
sample = pd.read_pickle(root_path / f"sample_T{str(nr_timesteps)}.pkl")

# Calculate covariance/sd
# 1. Calculate Mean
mean = sample.mean()
print("Mean:")
print(mean)
print("\n")
mean.to_excel(root_path / f"sample_mean{str(nr_timesteps)}.xlsx")


# 2. Calculate Standard Deviation
sd = sample.std()
print("Standard Deviation:")
print(sd)
print("\n")
sd.to_excel(root_path / f"sample_sd{str(nr_timesteps)}.xlsx")

# 3. Calculate Correlation Matrix
correlation = sample.corr()
correlation = round(correlation, 3)

# Alternative: if you want to remove based on actual correlation matrix values being identical
unique_rows = ~correlation.duplicated()
unique_cols = ~correlation.T.duplicated()
correlation_unique = correlation.loc[unique_rows, unique_cols]

print("Correlation Matrix:")
print(correlation)
print("\n")
correlation.to_excel(root_path / f"sample_correlation{str(nr_timesteps)}.xlsx")

correlation_unique.index = correlation_unique.index.get_level_values(1)
correlation_unique.columns = correlation_unique.columns.get_level_values(1)

correlation_unique.index = correlation_unique.index.map(lambda x: x[0] if isinstance(x, tuple) else x)
correlation_unique.columns = correlation_unique.columns.map(lambda x: x[0] if isinstance(x, tuple) else x)
correlation_unique.to_excel(root_path / f"sample_correlation_unique{str(nr_timesteps)}.xlsx")

correlation_input = pd.read_csv(root_path / "Crystal_Ball" / "mean_variance" / "technology_capex" / "correlation.csv",
                                index_col=0)

common_index = correlation_input.index.intersection(correlation_unique.index)
common_cols = correlation_input.columns.intersection(correlation_unique.columns)

# Filter both dataframes to only have common indices and columns
corr_input_filtered = correlation_input.loc[common_index, common_cols]
corr_unique_filtered = correlation_unique.loc[common_index, common_cols]

# Compute the difference in correlation coefficients
correlation_diff = corr_input_filtered - corr_unique_filtered
print(correlation_diff)

# Summary statistics of the differences
print("\nDifference Statistics:")
print(f"Mean difference: {correlation_diff.values.mean():.6f}")
print(f"Max absolute difference: {abs(correlation_diff.values).max():.6f}")
print(f"Min difference: {correlation_diff.values.min():.6f}")
print(f"Max difference: {correlation_diff.values.max():.6f}")

# Identify largest differences
diff_stacked = correlation_diff.stack()
largest_diffs = diff_stacked.abs().nlargest(20)
print("\nTop 10 largest absolute differences:")
print(largest_diffs)
correlation_diff.to_excel(root_path / f"sample_correlation_diff{str(nr_timesteps)}.xlsx")


# 4. Calculate Covariance Matrix
covariance = sample.cov()
print("Covariance Matrix:")
print(covariance)
covariance.to_excel(root_path / f"sample_covariance{str(nr_timesteps)}.xlsx")



#Calculate min/max costs
m_api = ModelApi(config=f"./config.json", dataset=dataset)
m_api.build_model()

capex_specific_conversion_original = m_api.capex_specific_conversion.copy()
capex_specific_conversion_original = capex_specific_conversion_original.rename(
    {
        old: new
        for old, new in zip(
        list(capex_specific_conversion_original.dims),
        [
            "set_conversion_technologies",
            "set_nodes",
            "set_time_steps_yearly",
        ],
        strict=False,
    )
    }
)

tech_dim = "set_conversion_technologies"
location_dim = "set_nodes"
techs = capex_specific_conversion_original.coords[tech_dim].values
nodes = m_api.optimization_setup.sets["set_nodes"]

for index, sample_row in sample.iterrows():
    tech_sample = sample_row["technology_capex"]
    delta_xr = generate_delta_xr_technologies(tech_sample, techs, capex_specific_conversion_original, tech_dim, location_dim)

    capex_specific_conversion = capex_specific_conversion_original + delta_xr

    print(capex_specific_conversion.values.min())
    print(capex_specific_conversion.values.max())


