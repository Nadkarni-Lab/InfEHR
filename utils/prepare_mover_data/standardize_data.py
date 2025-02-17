import os
import pandas as pd
import numpy as np
import logging
import joblib
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
from typing import Dict, Union, List

def setup_logger(log_dir):
    """Set up and return a configured logger"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file_path = os.path.join(log_dir, f'standardization_{timestamp}.log')
    
    file_handler = logging.FileHandler(log_file_path, mode='a')
    console_handler = logging.StreamHandler()
    file_handler.setLevel(logging.INFO)
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def process_blood_pressure(vitals_path: str, output_dir: str, logger) -> str:
    """Process blood pressure readings from vitals data"""
    logger.info("Starting BP processing")
    
    # Load vitals data
    flow_sheet = pd.read_csv(vitals_path)
    logger.info(f'Initial flow sheet length: {len(flow_sheet)}')
    
    # Filter BP rows
    bp_rows = flow_sheet[flow_sheet['FLO_DISPLAY_NAME'] == 'BP'].copy()
    logger.info(f'BP rows found: {len(bp_rows)}')
    
    # Split BP values and convert to numeric
    bp_rows[['Systolic', 'Diastolic']] = bp_rows['MEAS_VALUE'].str.split('/', expand=True)
    bp_rows['Systolic'] = pd.to_numeric(bp_rows['Systolic'], errors='coerce')
    bp_rows['Diastolic'] = pd.to_numeric(bp_rows['Diastolic'], errors='coerce')
    
    # Create separate records
    systolic_df = bp_rows[['FLO_DISPLAY_NAME', 'Systolic']].copy()
    systolic_df['FLO_DISPLAY_NAME'] = 'BLOOD PRESSURE - systolic'
    systolic_df = systolic_df.rename(columns={'Systolic': 'MEAS_VALUE'})
    
    diastolic_df = bp_rows[['FLO_DISPLAY_NAME', 'Diastolic']].copy()
    diastolic_df['FLO_DISPLAY_NAME'] = 'BLOOD PRESSURE - diastolic'
    diastolic_df = diastolic_df.rename(columns={'Diastolic': 'MEAS_VALUE'})
    
    # Remove original BP rows and combine
    flow_sheet = flow_sheet[flow_sheet['FLO_DISPLAY_NAME'] != 'BP']
    result_df = pd.concat([flow_sheet, systolic_df, diastolic_df], ignore_index=True)
    
    # Save processed data
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'vitals_with_bp_split.csv')
    result_df.to_csv(output_path, index=False)
    
    logger.info(f'BP processing complete. Output saved to: {output_path}')
    return output_path

class ClinicalLabStandardizer:
    """Handles standardization of clinical lab measurements"""
    def __init__(self):
        # Define standard units for each lab test
        self.standard_units = {
            'Albumin': 'g/dL',
            'Creatinine': 'mg/dL',
            'Troponin I.cardiac': 'ng/mL',
            'Leukocytes': 'K/µL',
            'Magnesium': 'mg/dL',
            'Protein': 'g/dL'
        }
        
        # Define conversion factors
        self.conversion_factors = {
            'Albumin': {
                'MG/L': lambda x: x / 10000,  # mg/L to g/dL
                'G/TOT VOL': lambda x: x      # already in g/dL
            },
            'Creatinine': {
                'G/TOT VOL': lambda x: x * 100  # g/dL to mg/dL
            }
            # Add more conversions as needed
        }
    
    def standardize_value(self, lab_name: str, value: float, original_unit: str) -> Dict:
        """Standardize a single lab value"""
        if lab_name not in self.standard_units:
            return {'value': value, 'unit': original_unit, 'converted': False}
            
        standard_unit = self.standard_units[lab_name]
        
        if lab_name in self.conversion_factors and original_unit in self.conversion_factors[lab_name]:
            converted_value = self.conversion_factors[lab_name][original_unit](value)
            return {'value': converted_value, 'unit': standard_unit, 'converted': True}
            
        return {'value': value, 'unit': original_unit, 'converted': False}

def process_lab_results(labs_path: str, output_dir: str, logger) -> str:
    """Process and standardize laboratory results"""
    logger.info("Starting lab results processing")
    
    # Load lab data
    labs_df = pd.read_csv(labs_path)
    standardizer = ClinicalLabStandardizer()
    
    # Process each lab result
    results = []
    for _, row in labs_df.iterrows():
        try:
            value = float(row['MEAS_VALUE'])
            standardized = standardizer.standardize_value(
                row['Lab Name'],
                value,
                row['Measurement Units']
            )
            row['standardized_value'] = standardized['value']
            row['standardized_unit'] = standardized['unit']
            results.append(row)
        except (ValueError, TypeError):
            continue
    
    result_df = pd.DataFrame(results)
    
    # Save processed data
    output_path = os.path.join(output_dir, 'standardized_labs.csv')
    result_df.to_csv(output_path, index=False)
    
    logger.info(f'Lab processing complete. Output saved to: {output_path}')
    return output_path

def process_vital_signs(vitals_path: str, output_dir: str, logger) -> str:
    """Process and clean vital signs data"""
    logger.info("Starting vital signs processing")
    
    # Load vitals data
    vitals_df = pd.read_csv(vitals_path)
    
    # Calculate thresholds for each vital sign
    vital_stats = {}
    for vital in vitals_df['FLO_DISPLAY_NAME'].unique():
        data = pd.to_numeric(vitals_df[vitals_df['FLO_DISPLAY_NAME'] == vital]['MEAS_VALUE'], 
                           errors='coerce')
        vital_stats[vital] = {
            'p1': data.quantile(0.01),
            'p99': data.quantile(0.99)
        }
    
    # Clean data using thresholds
    cleaned_data = []
    for _, row in vitals_df.iterrows():
        try:
            value = float(row['MEAS_VALUE'])
            vital = row['FLO_DISPLAY_NAME']
            if vital in vital_stats:
                if value >= vital_stats[vital]['p1'] and value <= vital_stats[vital]['p99']:
                    cleaned_data.append(row)
        except (ValueError, TypeError):
            continue
    
    result_df = pd.DataFrame(cleaned_data)
    
    # Create and save histograms
    plot_dir = os.path.join(output_dir, 'histograms')
    os.makedirs(plot_dir, exist_ok=True)
    
    for vital in result_df['FLO_DISPLAY_NAME'].unique():
        data = pd.to_numeric(result_df[result_df['FLO_DISPLAY_NAME'] == vital]['MEAS_VALUE'], 
                           errors='coerce')
        plt.figure(figsize=(10, 6))
        plt.hist(data, bins=50)
        plt.title(f'Distribution of {vital}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(plot_dir, f'{vital}_histogram.png'))
        plt.close()
    
    # Save processed data
    output_path = os.path.join(output_dir, 'cleaned_vitals.csv')
    result_df.to_csv(output_path, index=False)
    
    logger.info(f'Vital signs processing complete. Output saved to: {output_path}')
    return output_path

def parse_arguments():
    parser = argparse.ArgumentParser(description='Standardize clinical data.')
    
    parser.add_argument('--vitals_path', required=True,
                        help='Path to vitals data CSV')
    parser.add_argument('--labs_path', required=True,
                        help='Path to laboratory results CSV')
    parser.add_argument('--output_dir', required=True,
                        help='Directory for output files')
    parser.add_argument('--log_dir', required=True,
                        help='Directory for log files')
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()
    
    # Setup logger
    logger = setup_logger(args.log_dir)
    
    try:
        # Process blood pressure
        bp_output = process_blood_pressure(args.vitals_path, args.output_dir, logger)
        
        # Process lab results
        labs_output = process_lab_results(args.labs_path, args.output_dir, logger)
        
        # Process vital signs (using BP-processed data)
        vitals_output = process_vital_signs(bp_output, args.output_dir, logger)
        
        logger.info("All processing completed successfully")
        
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
