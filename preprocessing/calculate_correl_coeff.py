import pandas as pd
import numpy as np
from pathlib import Path
import ast
from openpyxl import load_workbook
from openpyxl.formatting.rule import ColorScaleRule

# =============================================================================
# INPUT FILES
# =============================================================================
root_path =  Path("C:/Users/jwiegner/OneDrive - ETH Zurich/00_Papers_Journal/00_2026-Quadratic terms in ESM")
cost_data_path = root_path / Path("ZEN_garden_assumptions/assumptions_technologies_cost_uncertainty.xlsx")
technology_data_path = root_path / Path("ZEN_garden_assumptions/raw/assumptions_technologies_full.xlsx")
PRIMARY_WEIGHT = 0.6
SECONDARY_WEIGHT = 0.3
MIN_CORRELATION = 0.0
MAX_NUMBER_OF_SIMILAR_CARRIERS = 5

# options
use_input_output_data_from_zen_garden = True
use_number_of_carriers_only = True


# =============================================================================
# READ DATA
# =============================================================================

cost_df = pd.read_excel(
    cost_data_path,
    header=[0],
    index_col=0,
    sheet_name='Classification'
)

technology_df = pd.read_excel(
    technology_data_path,
    header=[0, 1],
    index_col=0
)

# =============================================================================
# 1) ADD COLUMN LEVEL "Classification"
# =============================================================================

cost_df.columns = pd.MultiIndex.from_product(
    [cost_df.columns, ["Classification"]]
)
data_df = pd.merge(cost_df, technology_df, left_index=True, right_index=True)

if use_input_output_data_from_zen_garden:
    data_df[('input_carrier', 'value')] = data_df[('input_carrier', 'value')].fillna(data_df[('reference_carrier', 'value')]).apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    data_df[('output_carrier', 'value')] = data_df[('output_carrier', 'value')].fillna(data_df[('reference_carrier', 'value')]).apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    data_df[('Secondary technology class', 'Classification')] = data_df[('input_carrier', 'value')] + data_df[('output_carrier', 'value')]
else:
    data_df[('Secondary technology class', 'Classification')] = (
        data_df[('Secondary technology class (input)', 'Classification')].fillna('').apply(lambda x: [x] if x != '' else [])
        +
        data_df[('Secondary technology class (output)', 'Classification')].fillna('').apply(lambda x: [x] if x != '' else [])
    )


# =============================================================================
# EXTRACT TECHNOLOGY CLASSIFICATION DATA
# =============================================================================
# Select relevant classification columns
classification_cols = [
    ('Primary technology class', 'Classification'),
    ('Secondary technology class', 'Classification'),
]

classification_df = data_df[classification_cols].copy()

# Export to latex
to_latex = data_df[classification_cols + [('TRL', 'Classification')]].copy()
col = ('Secondary technology class', 'Classification')
to_latex[col] = to_latex[col].apply(
    lambda x: sorted(set(x)) if isinstance(x, (list, set)) else x
)
to_latex.columns = to_latex.columns.droplevel(1)
to_latex = to_latex.rename(
    columns={
        to_latex.columns[1]: "Carriers and Commodities"
    }
)

to_latex['Carriers and Commodities'] = to_latex['Carriers and Commodities'].apply(
    lambda x: ", ".join(x) if isinstance(x, list) else x
).str.replace("_", " ")
to_latex.index = to_latex.index.str.replace("_", " ")

# Create LaTeX itemize list of unique primary technology classes
unique_classes = sorted(
    to_latex["Primary technology class"].dropna().unique()
)
latex_itemize = "\\begin{itemize}\n"
for c in unique_classes:
    latex_itemize += f"    \\item {c}\n"
latex_itemize += "\\end{itemize}"

print(latex_itemize)
with open(root_path / "latex_input" / "primary_technology_classes.tex", "w") as f:
    f.write(latex_itemize)

latex_table = to_latex.to_latex(
    index=True,
    escape=False,
    longtable=True,
    caption="Technology classifications.",
    label="tab:technology_classification"
)

print(latex_table)

# Optionally save
with open(root_path / "latex_input" / "technology_classification_table.tex", "w") as f:
    f.write(latex_table)


technologies = cost_df.index
# =============================================================================
# USER-DEFINED WEIGHTS
# =============================================================================
# Optional normalization
# weight_sum = (
#     PRIMARY_WEIGHT
#     + SECONDARY_WEIGHT
# )
#
# PRIMARY_WEIGHT /= weight_sum
# SECONDARY_WEIGHT /= weight_sum

# =============================================================================
# 2) CREATE EMPTY CORRELATION MATRIX
# =============================================================================
correlation_matrix = pd.DataFrame(
    0.0,
    index=technologies,
    columns=technologies
)

# =============================================================================
# 4) CALCULATE PAIRWISE WEIGHTED KEYWORD SIMILARITY
# =============================================================================
for tech_i in technologies:
    for tech_j in technologies:
        # Skip diagonal
        if tech_i == tech_j:
            continue
        score = 0.0
        # ---------------------------------------------------------------------
        # Primary technology class
        # ---------------------------------------------------------------------
        if classification_df.loc[tech_i, 'Primary technology class'][0] == "Other":
            pass
        elif (
            classification_df.loc[tech_i, 'Primary technology class'][0]
            ==
            classification_df.loc[tech_j, 'Primary technology class'][0]
        ):
            score += PRIMARY_WEIGHT

        # ---------------------------------------------------------------------
        # Secondary input class
        # ---------------------------------------------------------------------
        print(tech_i, tech_j)
        all_carriers = set(classification_df.loc[tech_i, 'Secondary technology class'][0] + classification_df.loc[tech_j, 'Secondary technology class'][0])
        carriers_present_in_both = set(classification_df.loc[tech_i, 'Secondary technology class'][0]) & set(classification_df.loc[tech_j, 'Secondary technology class'][0])

        print(len(carriers_present_in_both))

        if len(carriers_present_in_both) != 0:
            secondary_score = (0.5 + 0.1 * len(carriers_present_in_both))*SECONDARY_WEIGHT

            score += secondary_score
        #
        # if use_number_of_carriers_only:
        #     print(len(carriers_present_in_both))
        #     print(len(carriers_present_in_both)/MAX_NUMBER_OF_SIMILAR_CARRIERS)
        #     print(len(carriers_present_in_both)/MAX_NUMBER_OF_SIMILAR_CARRIERS * SECONDARY_WEIGHT)
        #     score += len(carriers_present_in_both)/MAX_NUMBER_OF_SIMILAR_CARRIERS * SECONDARY_WEIGHT
        # else:
        #     score += len(carriers_present_in_both) / len(all_carriers) * SECONDARY_WEIGHT
        print(score)

        correlation_matrix.loc[tech_i, tech_j] = score

# =============================================================================
# OPTIONAL: ADD MINIMUM BACKGROUND CORRELATION
# =============================================================================
correlation_matrix = (
    MIN_CORRELATION
    + (1 - MIN_CORRELATION) * correlation_matrix
)

# Restore diagonal to exactly 1
np.fill_diagonal(correlation_matrix.values, 1.0)

# =============================================================================
# RESULT
# =============================================================================

output_file = root_path / "ZEN_garden_assumptions" / "corelation_matrix.xlsx"

correlation_matrix.to_excel(output_file)

# =============================================================================
# LOAD WORKBOOK
# =============================================================================

wb = load_workbook(output_file)
ws = wb.active

# =============================================================================
# DETERMINE DATA RANGE
# =============================================================================

nrows = ws.max_row
ncols = ws.max_column

# Exclude index/column headers
data_range = f"B2:{ws.cell(nrows, ncols).coordinate}"

# =============================================================================
# CONDITIONAL FORMATTING
# =============================================================================

rule = ColorScaleRule(
    start_type='num',
    start_value=0,
    start_color='63BE7B',   # green

    mid_type='num',
    mid_value=0.5,
    mid_color='FFEB84',     # yellow

    end_type='num',
    end_value=1,
    end_color='F8696B'      # red
)

ws.conditional_formatting.add(data_range, rule)

# =============================================================================
# SAVE
# =============================================================================

wb.save(output_file)