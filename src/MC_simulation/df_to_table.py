import pandas as pd


def df_to_latex(df, table_name:str, table_label:str='my_table'):
    """
    Converts a Pandas DataFrame into a LaTeX table format as a string.
    """
    df = df.round(2)
    # Start the LaTeX table
    latex_str = "\\begin{table}[h]\n\\centering\n\\begin{tabular}{" + " | ".join(
        ["c"] * (df.shape[1] + 1)) + " }\n\\hline\n"

    # Add column headers
    latex_str += " & " + " & ".join(df.columns.astype(str)) + " \\\\\n\\hline\n"

    # Add rows dynamically
    for index, row in df.iterrows():
        latex_str += str(index) + " & " + " & ".join(row.astype(str)) + " \\\\\n"

    # Close the LaTeX table
    latex_str += f"\\hline\n\\end{{tabular}}\n\\caption{'{'}{table_name}{'}'}\n\\label{'{tab:'}{table_label}{'}'}\n\\end{'{table}'}"

    return latex_str.replace('_', ' ')

