import pandas as pd
from pathlib import Path

# Path to the Excel file
excel_path = Path("src/spring_datasheet_detection.xlsx")

# Read the Excel file
xls = pd.ExcelFile(excel_path)

# Iterate through each sheet
for sheet_name in xls.sheet_names:
    if sheet_name == "README":
        continue
    df = pd.read_excel(xls, sheet_name=sheet_name)
    print(f"Sheet: {sheet_name}")
    print("Columns:", df.columns.tolist())
    print("Ground Truth Column (C):")
    if 'ground_truth' in df.columns:
        gt_values = df['ground_truth'].tolist()
        print(gt_values)
        # Check if any are empty (including NaN)
        empty_count = sum(1 for v in gt_values if pd.isna(v) or v == '')
        print(f"Empty ground truth entries: {empty_count}")
    else:
        print("No 'ground_truth' column found")
    print("-" * 50)
