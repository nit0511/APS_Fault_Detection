from distutils import dir_util
from sensor.constant.training_pipeline import SCHEMA_FILE_PATH
from sensor.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from sensor.entity.config_entity import DataValidationConfig
from sensor.exception import SensorException
from sensor.logger import logging
from sensor.utils.main_utils import read_yaml_file, write_yaml_file
from scipy.stats import ks_2samp
import pandas as pd
import os, sys

class DataValidation:

    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_config: DataValidationConfig):
        try:
            logging.info("Initializing DataValidation class.")
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            logging.info("Schema file loaded successfully.")
        except Exception as e:
            logging.error(f"Error in DataValidation initialization: {e}")
            raise SensorException(e, sys)

    def drop_zero_std_columns(self, dataframe):
        pass

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            number_of_columns = len(self._schema_config["columns"])
            logging.info(f"Required number of columns: {number_of_columns}")
            logging.info(f"Data frame has {len(dataframe.columns)} columns.")
            
            if len(dataframe.columns) == number_of_columns:
                logging.info("Column validation passed.")
                return True
            
            logging.warning("Column validation failed.")
            return False
        except Exception as e:
            logging.error(f"Error in validate_number_of_columns: {e}")
            raise SensorException(e, sys)

    def is_numerical_column_exist(self, dataframe: pd.DataFrame) -> bool:
        try:
            numerical_columns = self._schema_config["numerical_columns"]
            dataframe_columns = dataframe.columns

            numerical_column_present = True
            missing_numerical_columns = []
            
            for num_column in numerical_columns:
                if num_column not in dataframe_columns:
                    numerical_column_present = False
                    missing_numerical_columns.append(num_column)
            
            if missing_numerical_columns:
                logging.warning(f"Missing numerical columns: {missing_numerical_columns}")
            else:
                logging.info("All numerical columns are present.")
            
            return numerical_column_present
        except Exception as e:
            logging.error(f"Error in is_numerical_column_exist: {e}")
            raise SensorException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            logging.info(f"Reading data from {file_path}")
            return pd.read_csv(file_path)
        except Exception as e:
            logging.error(f"Error reading data from {file_path}: {e}")
            raise SensorException(e, sys)
    
    def detect_dataset_drift(self, base_df, current_df, threshold=0.05) -> bool:
        try:
            logging.info("Detecting dataset drift...")
            status = True
            report = {}
            
            for column in base_df.columns:
                d1 = base_df[column]
                d2 = current_df[column]
                is_same_dist = ks_2samp(d1, d2)
                
                if threshold <= is_same_dist.pvalue:
                    is_found = False
                else:
                    is_found = True 
                    status = False
                
                report.update({column: {
                    "p_value": float(is_same_dist.pvalue),
                    "drift_status": is_found
                }})
                
                logging.info(f"Column: {column}, p-value: {is_same_dist.pvalue}, Drift detected: {is_found}")
            
            drift_report_file_path = self.data_validation_config.drift_report_file_path
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path, exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path, content=report)
            logging.info(f"Drift report saved at {drift_report_file_path}")
            
            return status
        except Exception as e:
            logging.error(f"Error in detect_dataset_drift: {e}")
            raise SensorException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logging.info("Starting data validation...")
            error_message = ""
            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            train_dataframe = DataValidation.read_data(train_file_path)
            test_dataframe = DataValidation.read_data(test_file_path)
            
            logging.info("Validating number of columns...")
            status = self.validate_number_of_columns(dataframe=train_dataframe)
            if not status:
                error_message += "Train dataframe does not contain all columns.\n"
            
            status = self.validate_number_of_columns(dataframe=test_dataframe)
            if not status:
                error_message += "Test dataframe does not contain all columns.\n"
            
            logging.info("Validating numerical columns...")
            status = self.is_numerical_column_exist(dataframe=train_dataframe)
            if not status:
                error_message += "Train dataframe does not contain all numerical columns.\n"
            
            status = self.is_numerical_column_exist(dataframe=test_dataframe)
            if not status:
                error_message += "Test dataframe does not contain all numerical columns.\n"
            
            if error_message:
                logging.error(f"Data validation errors: {error_message}")
                raise Exception(error_message)
            
            logging.info("Checking for data drift...")
            status = self.detect_dataset_drift(base_df=train_dataframe, current_df=test_dataframe)
            
            data_validation_artifact = DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=self.data_ingestion_artifact.trained_file_path,
                valid_test_file_path=self.data_ingestion_artifact.test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )
            
            logging.info(f"Data validation completed successfully: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            logging.error(f"Error in initiate_data_validation: {e}")
            raise SensorException(e, sys)

