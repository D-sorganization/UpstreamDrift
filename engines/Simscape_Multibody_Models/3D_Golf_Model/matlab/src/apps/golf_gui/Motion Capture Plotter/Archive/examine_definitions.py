import pandas as pd

# Read the Excel file
filename = "Wiffle_ProV1_club_3D_data.xlsx"
excel_file = pd.ExcelFile(filename)

print(f"Available sheets: {excel_file.sheet_names}")

# Examine the Definitions sheet
if "Definitions" in excel_file.sheet_names:
    print("\n=== Examining Definitions Sheet ===")
    df_def = pd.read_excel(filename, sheet_name="Definitions", header=None)

    print(f"Definitions sheet shape: {df_def.shape}")
    print(f"Definitions sheet columns: {len(df_def.columns)}")
    print(f"Definitions sheet rows: {len(df_def)}")

    print("\nFirst 20 rows of Definitions sheet:")
    print(df_def.head(20))

    print("\nAll content in Definitions sheet:")
    for i in range(len(df_def)):
        row = df_def.iloc[i]
        for j in range(len(row)):
            val = row[j]
            if pd.notna(val) and str(val).strip():
                print(f"Row {i}, Col {j}: '{val}'")
else:
    print("No Definitions sheet found!")

# Also check if there are any other sheets that might contain definitions
for sheet_name in excel_file.sheet_names:
    if "def" in sheet_name.lower() or "info" in sheet_name.lower():
        print(f"\n=== Examining {sheet_name} ===")
        df_info = pd.read_excel(filename, sheet_name=sheet_name, header=None)
        print(f"Shape: {df_info.shape}")
        print("First 10 rows:")
        print(df_info.head(10))
