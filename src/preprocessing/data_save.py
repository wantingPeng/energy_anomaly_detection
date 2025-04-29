import os
from src.utils.logger import logger
import dask.dataframe as dd
from datetime import datetime

def data_save(df, filename, save_dir=None):
    """
    Save DataFrame to parquet format using Dask.
    
    Args:
        df: Dask DataFrame or pandas DataFrame to save
        filename (str): Name of the file to save (without extension)
        save_dir (str, optional): Directory to save the file. If None, uses config default
    """
    try:
        # Convert to Dask DataFrame if it's pandas DataFrame
        if not isinstance(df, dd.DataFrame):
            df = dd.from_pandas(df, npartitions=1)
            
        # If save_dir not provided, load from config
        if save_dir is None:
            import yaml
            with open('configs/preprocessing.yaml', 'r') as file:
                config = yaml.safe_load(file)
                save_dir = config['data']['interim_dir']
        
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Generate full path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_path = os.path.join(save_dir, f"{filename}_{timestamp}.parquet")
        
        # Save the DataFrame
        logger.info(f"Saving data to: {full_path}")
        df.to_parquet(full_path)
        logger.info("Data saved successfully")
        
        return full_path
        
    except Exception as e:
        logger.error(f"Error saving data: {str(e)}")
        raise
