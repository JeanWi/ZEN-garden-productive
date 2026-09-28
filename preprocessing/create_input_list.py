from pathlib import Path
import pandas as pd
import json
import numpy as np

def get_attributes_from_dir(dir, skip_retrofitting_technologies=True):
    data = {}

    for subfolder in dir.iterdir():
        if "set_retrofitting_technologies" in str(subfolder) and skip_retrofitting_technologies:
            continue

        if subfolder.is_dir():
            attr_file = subfolder / "attributes.json"

            with open(attr_file, "r") as f:
                attributes = json.load(f)

            row = {}

            for attr_name, attr_values in attributes.items():
                # Extract values
                if isinstance(attr_values, dict):
                    default = attr_values.get("default_value")
                    unit = attr_values.get("unit")

                    # Handle "inf" string → np.inf
                    if default == "inf":
                        default = np.inf
                elif isinstance(attr_values, list):
                    for item in attr_values:
                        default = item.get("default_value")
                        unit = item.get("unit")

                        # Handle "inf" string → np.inf
                        if default == "inf":
                            default = np.inf

                row[(attr_name, "value")] = default
                row[(attr_name, "unit")] = unit

            for csv_file in subfolder.glob("*.csv"):
                csv_name = csv_file.stem  # filename without .csv

                # Option 1: just mark existence
                row[("variation", csv_name)] = 1

            data[subfolder.name] = row

    # Create DataFrame
    df = pd.DataFrame.from_dict(data, orient="index")


    # Create proper MultiIndex columns
    df.columns = pd.MultiIndex.from_tuples(df.columns)

    # Optional: sort columns nicely
    df = df.sort_index(axis=1)
    return df

def filter_eur_columns(df):

    units = df.xs("unit", level=1, axis=1)
    mask = units.apply(
        lambda col: col.astype(str).str.contains("EUR|Euro|€", case=False, na=False)
    ).any()
    attrs_with_eur = mask[mask].index
    df_eur = df.loc[:, df.columns.get_level_values(0).isin(attrs_with_eur)]
    return df_eur

carrier_path = Path("C:/ZenGardenInput/ZEN-models/data/Crystal_Ball/set_carriers")
df_carrier = get_attributes_from_dir(carrier_path)
df_carrier.to_excel("assumptions_carriers_full.xlsx")

df_carrier_cost = filter_eur_columns(df_carrier)
df_carrier_cost.to_excel("assumptions_carriers_cost.xlsx")


df_technology = []
for tec_set in ["set_conversion_technologies", "set_storage_technologies", "set_transport_technologies"]:
    technology_path = Path(f"C:/ZenGardenInput/ZEN-models/data/Crystal_Ball/set_technologies/{tec_set}")
    df_technology.append(get_attributes_from_dir(technology_path))

technology_path = Path(f"C:/ZenGardenInput/ZEN-models/data/Crystal_Ball/set_technologies/set_conversion_technologies/set_retrofitting_technologies")
df_technology.append(get_attributes_from_dir(technology_path, skip_retrofitting_technologies=False))


df_technology = pd.concat(df_technology, axis=0)
df_technology.to_excel("assumptions_technologies_full.xlsx")

df_technology_cost = filter_eur_columns(df_technology)
df_technology_cost.to_excel("assumptions_technologies_cost.xlsx")


