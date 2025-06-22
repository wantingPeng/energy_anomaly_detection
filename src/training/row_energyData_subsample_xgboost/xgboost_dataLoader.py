#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import yaml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path

from src.utils.logger import logger


class XGBoostDataLoader:
    """
    Data loader for XGBoost anomaly detection.
    This class handles loading and preprocessing data for XGBoost models.
    """
    
    def __init__(self, config_path):
        """
        Initialize the data loader with configuration.
        
        Args:
            config_path (str): Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self.train_data_path = self.config['data_config']['train_data']
        self.val_data_path = self.config['data_config']['val_data']
        self.test_data_path = self.config['data_config']['test_data']
        self.target_column = self.config['data_config']['target_column']
        self.save_dir = self.config['data_config']['save_dir']
        
        # Create save directory if it doesn't exist
        os.makedirs(Path(self.save_dir), exist_ok=True)
        
        # Initialize scaler
        self.scaler = StandardScaler()
        
    def _load_config(self, config_path):
        """
        Load configuration from YAML file.
        
        Args:
            config_path (str): Path to configuration file
            
        Returns:
            dict: Configuration dictionary
        """
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    
    def _preprocess_data(self, df):
        """
        Preprocess the data by handling missing values, etc.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: Preprocessed DataFrame
        """
        # Fill missing values with column mean
        for col in df.columns:
            if df[col].isna().sum() > 0:
                if col != self.target_column:
                    df[col] = df[col].fillna(df[col].mean())
                else:
                    df[col] = df[col].fillna(0)  # Assume non-anomalous if target is missing
                    
        # Log preprocessing results
        logger.info(f"Preprocessing completed. DataFrame shape: {df.shape}")
        
        return df
    
    def load_train_data(self):
        """
        Load and preprocess training data.
        
        Returns:
            tuple: X_train, y_train - features and target variables
        """
        logger.info(f"Loading training data from {self.train_data_path}")
        try:
            df_train = pd.read_parquet(self.train_data_path)
            logger.info(f"Training data loaded successfully. Shape: {df_train.shape}")
            
            # Preprocess data
            df_train = self._preprocess_data(df_train)
            
            # Split features and target
            X_train = df_train.drop(columns=[self.target_column])
            y_train = df_train[self.target_column]
            
            # Save column names for later use
            self.feature_names = X_train.columns.tolist()
            
            # Log class distribution
            logger.info(f"Training data class distribution: \n{y_train.value_counts()}")
            
            return X_train, y_train
            
        except Exception as e:
            logger.error(f"Error loading training data: {str(e)}")
            raise
    
    def load_val_data(self):
        """
        Load and preprocess validation data.
        
        Returns:
            tuple: X_val, y_val - features and target variables
        """
        logger.info(f"Loading validation data from {self.val_data_path}")
        try:
            df_val = pd.read_parquet(self.val_data_path)
            logger.info(f"Validation data loaded successfully. Shape: {df_val.shape}")
            
            # Preprocess data
            df_val = self._preprocess_data(df_val)
            
            # Split features and target
            X_val = df_val.drop(columns=[self.target_column])
            y_val = df_val[self.target_column]
            
            # Log class distribution
            logger.info(f"Validation data class distribution: \n{y_val.value_counts()}")
            
            return X_val, y_val
            
        except Exception as e:
            logger.error(f"Error loading validation data: {str(e)}")
            raise
    
    def load_test_data(self):
        """
        Load and preprocess test data.
        
        Returns:
            tuple: X_test, y_test - features and target variables
        """
        logger.info(f"Loading test data from {self.test_data_path}")
        try:
            df_test = pd.read_parquet(self.test_data_path)
            logger.info(f"Test data loaded successfully. Shape: {df_test.shape}")
            
            # Preprocess data
            df_test = self._preprocess_data(df_test)
            
            # Split features and target
            X_test = df_test.drop(columns=[self.target_column])
            y_test = df_test[self.target_column]
            
            # Log class distribution
            logger.info(f"Test data class distribution: \n{y_test.value_counts()}")
            
            return X_test, y_test
            
        except Exception as e:
            logger.error(f"Error loading test data: {str(e)}")
            raise
    
    def get_feature_names(self):
        """
        Get the feature names.
        
        Returns:
            list: List of feature names
        """
        return self.feature_names
