import pandas as pd
from src.utils.logger import logger


def calculate_component_proportions(parquet_file_path):
    try:
        # Read the parquet file
        df = pd.read_parquet(parquet_file_path)
        
        # Calculate the total number of components
        total_components = len(df)
        
        # Calculate the proportion of each component type
        proportions = {}
        for component in ['component_type_contact', 'component_type_pcb', 'component_type_ring']:
            count = df[component].sum()
            proportions[component] = count / total_components
            
        # Log the proportions
        logger.info(f"Component proportions: {proportions}")
        
        return proportions
    except Exception as e:
        logger.error(f"Error calculating component proportions: {e}")
        return None

# Example usage
if __name__ == "__main__":
    parquet_file_path = "Data/processed/lsmt/merged/test.parquet"
    calculate_component_proportions(parquet_file_path)
