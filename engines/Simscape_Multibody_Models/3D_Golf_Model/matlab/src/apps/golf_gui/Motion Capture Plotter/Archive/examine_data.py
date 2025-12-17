import pandas as pd

# Read the Excel file
filename = "Wiffle_ProV1_club_3D_data.xlsx"
excel_file = pd.ExcelFile(filename)

print(f"Available sheets: {excel_file.sheet_names}")

# Examine the first sheet
sheet_name = "TW_wiffle"
df = pd.read_excel(filename, sheet_name=sheet_name, header=None)

print(f"\n=== Examining {sheet_name} ===")
print(f"Shape: {df.shape}")
print(f"Columns: {len(df.columns)}")
print(f"Rows: {len(df)}")

# Show first few rows
print("\nFirst 5 rows:")
print(df.head())

# Show column headers if they exist
print("\nColumn 0 (first column):")
print(df.iloc[0])

print("\nColumn 1 (second column):")
print(df.iloc[1])

# Check for any non-numeric data
print("\nChecking for text/non-numeric data in first few rows:")
for i in range(min(5, len(df))):
    for j in range(min(20, len(df.columns))):
        val = df.iloc[i, j]
        if isinstance(val, str):
            print(f"Row {i}, Col {j}: '{val}' (string)")
        elif pd.isna(val):
            print(f"Row {i}, Col {j}: NaN")
        else:
            print(f"Row {i}, Col {j}: {val} ({type(val)})")
    print("---")
