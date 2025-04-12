""" Functions that turn dataframes into simple LaTeX tables stored as txt files."""

def df_to_latex(df, table_name:str, table_label:str='my_table', round=4):
    """
    Converts a Pandas DataFrame into a LaTeX table format as a string.
    """
    df = df.round(round)
    df = df.apply(lambda col: col.map(lambda x: f"{x:.{round}f}" if isinstance(x, (int, float)) else x))
    latex_str = "\\begin{table}[h]\n\\centering\n\\begin{tabular}{" + " ".join(
        ["c"] * (df.shape[1] + 1)) + " }\n\\hline\n"

    latex_str += " & " + " & ".join(df.columns.astype(str)) + " \\\\\n\\hline\n"

    for index, row in df.iterrows():
        latex_str += str(index) + " & " + " & ".join(row.astype(str)) + " \\\\\n"

    latex_str += f"\\hline\n\\end{{tabular}}\n\\caption{'{'}{table_name}{'}'}\n\\label{'{tab:'}{table_label}{'}'}\n\\end{'{table}'}"

    return latex_str.replace('_', ' ')


def df_to_latex_regression_table(ols_df):
    """
    Converts a DataFrame to a table displaying regression coefficients and standard errors.
    """
    latex_table = """
        \\begin{table}[ht]
        \\centering
        \\begin{tabular}{l c}
        \\hline
        \\hline
        Variable & Coefficient \\\\
        \\midrule
    """

    for index, row in ols_df.iterrows():
        variable = index
        coefficient = row['Coefficient']
        std_error = row['Std Error']
        latex_table += f"{variable} & {coefficient:.4f} \\\\ \n"
        latex_table += f"      & \\scriptsize({std_error:.4f}) \\\\ \n"  # std_error is now smaller

    latex_table += """
        \\bottomrule
        \\end{tabular}
        \\caption{OLS Regression Results}
        \\end{table}
    """

    return latex_table


def df_to_latex_regression_table_with_two_columns(ols_df, table_name, macro, no_macro):
    """
    Converts a DataFrame to a table with two columns displaying regression coefficients and standard errors.
    """
    latex_table = """
        \\begin{table}[ht]
        \\centering
        \\begin{tabular}{l c c}
        \\hline
        Variable & Model 1 & Model 2 \\\\
        \\hline
    """

    for index, row in ols_df.iterrows():
        variable = index
        coef_ols1 = row[f'{no_macro} Coefficient']
        std_error_ols1 = row[f'{no_macro} Std Error']
        coef_ols2 = row[f'{macro} Coefficient']
        std_error_ols2 = row[f'{macro} Std Error']

        latex_table += f"{variable} & {coef_ols1:.4f} & {coef_ols2:.4f} \\\\ \n"
        latex_table += f"         & \\scriptsize({std_error_ols1:.4f}) & \\scriptsize({std_error_ols2:.4f}) \\\\ \n"

    latex_table += """
        \\hline
        \\end{tabular}
        \\caption{""" + table_name + """}
        \\end{table}
        """

    return latex_table
