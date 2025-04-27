import dask.dataframe as dd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import os
from datetime import datetime

def analyze_column_quality(df: pd.DataFrame, column: str) -> dict:
    """Analyze the quality of a single column."""
    data = df[column]
    n_total = len(data)
    n_missing = data.isna().sum()
    n_zeros = (data == 0).sum()
    n_unique = data.nunique()
    variance = data.var()
    
    return {
        'total_count': n_total,
        'missing_count': n_missing,
        'missing_percentage': (n_missing / n_total) * 100,
        'zero_count': n_zeros,
        'zero_percentage': (n_zeros / n_total) * 100,
        'unique_values': n_unique,
        'has_variance': variance > 0
    }

def analyze_energy_data(data_path: str, output_dir: str, file_pattern: str = "Dezember_2024.csv"):
    """
    Analyze energy data using Dask and generate statistical summaries and visualizations.
    
    Args:
        data_path: Path to the parent directory containing Ring, PCB, and Contacting folders
        output_dir: Directory to save visualizations
        file_pattern: Pattern of files to analyze
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize results dictionary
    results = {}
    
    # Process each subdirectory
    for subdir in ['Ring', 'PCB', 'Contacting']:
        file_path = os.path.join(data_path, subdir, file_pattern)
        if not os.path.exists(file_path):
            print(f"Warning: File not found - {file_path}")
            continue
            
        print(f"\nProcessing {subdir} data...")
        
        # Read data using Dask with assume_missing=True to handle mixed types
        df = dd.read_csv(file_path, assume_missing=True)
        df_computed = df.compute()
        
        # Get numeric columns only
        numeric_cols = df_computed.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            print(f"Warning: No numeric columns found in {subdir} dataset")
            continue
            
        print(f"Found {len(numeric_cols)} numeric columns: {', '.join(numeric_cols)}")
        
        # Analyze data quality for each column
        print("\nData Quality Analysis:")
        quality_results = {}
        columns_with_variance = []
        
        for col in numeric_cols:
            quality = analyze_column_quality(df_computed, col)
            quality_results[col] = quality
            if quality['has_variance']:
                columns_with_variance.append(col)
            
            print(f"\n{col}:")
            print(f"  - Missing values: {quality['missing_count']} ({quality['missing_percentage']:.2f}%)")
            print(f"  - Zero values: {quality['zero_count']} ({quality['zero_percentage']:.2f}%)")
            print(f"  - Unique values: {quality['unique_values']}")
            print(f"  - Has variance: {'Yes' if quality['has_variance'] else 'No'}")
        
        # Calculate statistics for numeric columns only
        print(f"\nCalculating statistics for {subdir}...")
        stats_dict = {
            'mean': df_computed[numeric_cols].mean(),
            'median': df_computed[numeric_cols].median(),
            'std': df_computed[numeric_cols].std(),
            'min': df_computed[numeric_cols].min(),
            'max': df_computed[numeric_cols].max(),
            'q1': df_computed[numeric_cols].quantile(0.25),
            'q3': df_computed[numeric_cols].quantile(0.75)
        }
        
        results[subdir] = {
            'stats': stats_dict,
            'quality': quality_results
        }
        
        # Generate visualizations for columns with variance
        print(f"\nGenerating visualizations for {subdir}...")
        print(f"Note: Only plotting {len(columns_with_variance)} columns with non-zero variance")
        
        if len(columns_with_variance) > 0:
            # 1. Histogram
            n_cols = min(3, len(columns_with_variance))
            n_rows = (len(columns_with_variance) + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            axes = np.array(axes).reshape(-1)  # Flatten axes array
            
            for i, (ax, column) in enumerate(zip(axes, columns_with_variance)):
                ax.hist(df_computed[column].dropna(), bins=50, density=True, alpha=0.7)
                ax.set_title(f'{column}')
                ax.set_xlabel('Value')
                ax.set_ylabel('Density')
            
            # Hide empty subplots
            for ax in axes[len(columns_with_variance):]:
                ax.set_visible(False)
                
            plt.suptitle(f'Histograms - {subdir}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{subdir}_histogram.png'))
            plt.close()
            
            # 2. Boxplot
            plt.figure(figsize=(15, 8))
            sns.boxplot(data=df_computed[columns_with_variance])
            plt.title(f'Boxplot - {subdir}')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{subdir}_boxplot.png'))
            plt.close()
            
            # 3. KDE Plot
            plt.figure(figsize=(15, 8))
            for column in columns_with_variance:
                sns.kdeplot(data=df_computed[column].dropna(), label=column)
            plt.title(f'KDE Plot - {subdir}')
            plt.xlabel('Value')
            plt.ylabel('Density')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{subdir}_kde.png'))
            plt.close()
            
            # 4. Time Series Plot
            n_cols = min(2, len(columns_with_variance))
            n_rows = (len(columns_with_variance) + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            axes = np.array(axes).reshape(-1)  # Flatten axes array
            
            for i, (ax, column) in enumerate(zip(axes, columns_with_variance)):
                ax.plot(df_computed.index, df_computed[column], alpha=0.7)
                ax.set_title(f'{column}')
                ax.set_xlabel('Time')
                ax.set_ylabel('Value')
            
            # Hide empty subplots
            for ax in axes[len(columns_with_variance):]:
                ax.set_visible(False)
                
            plt.suptitle(f'Time Series Plots - {subdir}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{subdir}_timeseries.png'))
            plt.close()
            
            # 5. Q-Q Plot
            n_cols = min(3, len(columns_with_variance))
            n_rows = (len(columns_with_variance) + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            axes = np.array(axes).reshape(-1)  # Flatten axes array
            
            for i, (ax, column) in enumerate(zip(axes, columns_with_variance)):
                stats.probplot(df_computed[column].dropna(), dist="norm", plot=ax)
                ax.set_title(f'Q-Q Plot - {column}')
            
            # Hide empty subplots
            for ax in axes[len(columns_with_variance):]:
                ax.set_visible(False)
                
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{subdir}_qqplot.png'))
            plt.close()
        
        print(f"Completed processing {subdir}")
    
    # Generate markdown report
    print("\nGenerating final report...")
    generate_markdown_report(results, output_dir)
    print("Analysis complete! Check the ColumnStats.md file in the visualizations directory.")

def generate_markdown_report(results: dict, output_dir: str):
    """Generate a markdown report with the statistical results."""
    markdown_content = "# Column Statistics Report\n\n"
    markdown_content += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    for subdir, data in results.items():
        markdown_content += f"## {subdir}\n\n"
        
        # Data Quality Section
        markdown_content += "### Data Quality Analysis\n\n"
        markdown_content += "| Column | Missing Values | Zero Values | Unique Values | Has Variance |\n"
        markdown_content += "|--------|----------------|-------------|---------------|-------------|\n"
        
        for col, quality in data['quality'].items():
            markdown_content += (
                f"| {col} | {quality['missing_count']} ({quality['missing_percentage']:.2f}%) | "
                f"{quality['zero_count']} ({quality['zero_percentage']:.2f}%) | "
                f"{quality['unique_values']} | "
                f"{'Yes' if quality['has_variance'] else 'No'} |\n"
            )
        
        # Statistics Section
        markdown_content += "\n### Statistical Analysis\n\n"
        markdown_content += "| Statistic | Value |\n"
        markdown_content += "|-----------|-------|\n"
        
        for stat_name, stat_value in data['stats'].items():
            if isinstance(stat_value, pd.Series):
                for col, val in stat_value.items():
                    markdown_content += f"| {stat_name} ({col}) | {val:.4f} |\n"
            else:
                markdown_content += f"| {stat_name} | {stat_value:.4f} |\n"
        
        # Visualizations Section
        markdown_content += "\n### Visualizations\n\n"
        markdown_content += "Note: Visualizations only include columns with non-zero variance.\n\n"
        markdown_content += f"- ![Histogram]({subdir}_histogram.png)\n"
        markdown_content += f"- ![Boxplot]({subdir}_boxplot.png)\n"
        markdown_content += f"- ![KDE Plot]({subdir}_kde.png)\n"
        markdown_content += f"- ![Time Series]({subdir}_timeseries.png)\n"
        markdown_content += f"- ![Q-Q Plot]({subdir}_qqplot.png)\n\n"
    
    with open(os.path.join(output_dir, 'ColumnStats.md'), 'w') as f:
        f.write(markdown_content)

if __name__ == "__main__":
    data_path = "Data/Energy_Data"
    output_dir = "Data/visualizations"
    analyze_energy_data(data_path, output_dir) 