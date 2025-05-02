import pandas as pd
from sklearn.model_selection import train_test_split
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import PatternFill

def stratified_excel_split(input_file, output_file, 
                         feature_cols=['Polarity', 'Subjectivity'],
                         target_col='Mode',
                         test_size=0.7, random_state=42,
                         sheet_name=0):
    
    try:
        data=pd.read_excel(input_file, sheet_name=sheet_name)
    except Exception as e:
        raise ValueError(f"Error reading Excel file: {str(e)}")
    required_cols=feature_cols + [target_col]
    missing_cols=[col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    df=data[required_cols].copy()

    train_df, test_df=train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[target_col])
    train_df['Set_Type']='Training'
    test_df['Set_Type']='Test'

    marked_data=pd.concat([train_df, test_df])
    def get_distribution(series):
        return series.value_counts(normalize=True).to_dict()
    
    original_dist=get_distribution(df[target_col])
    train_dist=get_distribution(train_df[target_col])
    test_dist=get_distribution(test_df[target_col])
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        marked_data.to_excel(writer, sheet_name='Split_Data', index=False)
        summary = pd.DataFrame({
            'Class': list(original_dist.keys()),
            'Original %': [f"{v*100:.1f}%" for v in original_dist.values()],
            'Training %': [f"{train_dist.get(k, 0)*100:.1f}%" for k in original_dist.keys()],
            'Test %': [f"{test_dist.get(k, 0)*100:.1f}%" for k in original_dist.keys()],
        })
        summary.to_excel(writer, sheet_name='Distribution_Summary', index=False)
    print("\n" + "="*50)
    print("Stratified Split Complete")
    print("="*50)
    print(f"\nOriginal class distribution:")
    for cls, pct in original_dist.items():
        print(f"- {cls}: {pct*100:.1f}%")
    
    print(f"\nTraining set ({len(train_df)} rows):")
    for cls, pct in train_dist.items():
        print(f"- {cls}: {pct*100:.1f}%")
    
    print(f"\nTest set ({len(test_df)} rows):")
    for cls, pct in test_dist.items():
        print(f"- {cls}: {pct*100:.1f}%")
    
    print(f"\nSaved to: {output_file}")
    print("- 'Split_Data' sheet: Marked dataset")
    print("- 'Distribution_Summary' sheet: Class distribution report")

if __name__ == "__main__":
    input_file="/Users/tobylodge/Downloads/Training and Test Data Dissertation/genre_splits/testtrain_Blues.xlsx"  
    output_file="test_trainiing_songs_output_Blues.xlsx"
    
    stratified_excel_split(input_file=input_file, output_file=output_file, feature_cols=['Polarity', 'Subjectivity'], target_col='Mode', test_size=0.7, random_state=42)
