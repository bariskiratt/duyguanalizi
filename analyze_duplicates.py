import pandas as pd
import os
from pathlib import Path

def analyze_file(file_path):
    """Analyze a parquet file and return basic stats"""
    try:
        df = pd.read_parquet(file_path)
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        
        # Check for duplicates
        total_rows = len(df)
        unique_texts = df['review_text'].nunique() if 'review_text' in df.columns else total_rows
        duplicates = total_rows - unique_texts
        
        return {
            'file': file_path,
            'size_mb': round(file_size_mb, 2),
            'total_rows': total_rows,
            'unique_texts': unique_texts,
            'duplicates': duplicates,
            'duplicate_percentage': round((duplicates / total_rows * 100), 2) if total_rows > 0 else 0,
            'columns': list(df.columns)
        }
    except Exception as e:
        return {
            'file': file_path,
            'error': str(e)
        }

def main():
    print("🔍 Analyzing data files for duplicate issues...")
    print("="*80)
    
    # Analyze all parquet files in processed directory
    processed_dir = Path("data/processed")
    parquet_files = list(processed_dir.glob("*.parquet"))
    
    results = []
    for file_path in parquet_files:
        print(f"\n📊 Analyzing: {file_path.name}")
        result = analyze_file(file_path)
        results.append(result)
        
        if 'error' in result:
            print(f"   ❌ Error: {result['error']}")
        else:
            print(f"   📁 Size: {result['size_mb']} MB")
            print(f"   📊 Total rows: {result['total_rows']:,}")
            print(f"   🔍 Unique texts: {result['unique_texts']:,}")
            print(f"   🚫 Duplicates: {result['duplicates']:,} ({result['duplicate_percentage']}%)")
            print(f"   📋 Columns: {result['columns']}")
    
    # Summary
    print("\n" + "="*80)
    print("📈 SUMMARY")
    print("="*80)
    
    valid_results = [r for r in results if 'error' not in r]
    
    if valid_results:
        total_rows = sum(r['total_rows'] for r in valid_results)
        total_duplicates = sum(r['duplicates'] for r in valid_results)
        total_size = sum(r['size_mb'] for r in valid_results)
        
        print(f"📊 Total rows across all files: {total_rows:,}")
        print(f"🚫 Total duplicates: {total_duplicates:,}")
        print(f"📁 Total size: {total_size:.2f} MB")
        print(f"📈 Overall duplicate percentage: {(total_duplicates/total_rows*100):.2f}%")
        
        # Check merged.parquet specifically
        merged_result = next((r for r in valid_results if 'merged.parquet' in r['file']), None)
        if merged_result:
            print(f"\n🎯 MERGED.PARQUET ANALYSIS:")
            print(f"   📊 Rows: {merged_result['total_rows']:,}")
            print(f"   🚫 Duplicates: {merged_result['duplicates']:,}")
            print(f"   📈 Duplicate %: {merged_result['duplicate_percentage']}%")
            
            if merged_result['duplicates'] > 0:
                print(f"\n⚠️  DUPLICATE ISSUE DETECTED!")
                print(f"   Your peer has ~1M rows, you have {merged_result['total_rows']:,} rows")
                print(f"   This suggests {merged_result['duplicates']:,} duplicate entries")
                
                # Check if DROP_DUPLICATES is enabled
                print(f"\n💡 SOLUTION:")
                print(f"   Set DROP_DUPLICATES = True in merge_datasets.py")
                print(f"   This will remove {merged_result['duplicates']:,} duplicate entries")

if __name__ == "__main__":
    main()
