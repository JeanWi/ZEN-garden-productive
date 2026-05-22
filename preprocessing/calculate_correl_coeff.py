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
PRIMARY_CORRELATION = 0.9
SECONDARY_CORRELATION = 0.3
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

classification_df.columns = classification_df.columns.droplevel(1)
# Primary
primary_dummies = pd.get_dummies(classification_df["Primary technology class"]).astype(float)
primary_dummies = primary_dummies * PRIMARY_CORRELATION          # scale to correlation value

all_carriers = sorted({
    c
    for carriers in classification_df["Secondary technology class"]
    for c in (carriers if isinstance(carriers, list) else [])
})

# Secondary
secondary_dummies = pd.DataFrame(0.0, index=technologies, columns=all_carriers)
for tech in technologies:
    carriers = classification_df.loc[tech, "Secondary technology class"]
    if isinstance(carriers, list):
        for c in carriers:
            if c in secondary_dummies.columns:
                secondary_dummies.loc[tech, c] = SECONDARY_CORRELATION

classification_df_dummies = pd.concat([primary_dummies, secondary_dummies], axis=1)

B = classification_df_dummies.astype(float).values  # rows: technologies, cols: factors

# cosine similarity (row-normalized dot product)
norms = np.linalg.norm(B, axis=1, keepdims=True)
B_norm = np.divide(B, norms, where=norms != 0)

corr_mat = B_norm @ B_norm.T

# enforce symmetry + exact diagonal = 1
np.fill_diagonal(corr_mat, 1.0)
# lam = 1e-6
# corr_mat_pd = (1 - lam) * corr_mat + lam * np.eye(corr_mat.shape[0])

correlation_matrix = pd.DataFrame(
    corr_mat,
    index=classification_df_dummies.index,
    columns=classification_df_dummies.index
)
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