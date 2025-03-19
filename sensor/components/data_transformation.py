# import sys

# import numpy as np
# import pandas as pd
# from imblearn.combine import SMOTETomek
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import RobustScaler
# from sklearn.pipeline import Pipeline


# from sensor.constant.training_pipeline import TARGET_COLUMN
# from sensor.entity.artifact_entity import (
#     DataTransformationArtifact,
#     DataValidationArtifact,
# )
# from sensor.entity.config_entity import DataTransformationConfig
# from sensor.exception import SensorException
# from sensor.logger import logging
# from sensor.ml.model.estimator import TargetValueMapping
# from sensor.utils.main_utils import save_numpy_array_data, save_object




# class DataTransformation:
#     def __init__(self,data_validation_artifact: DataValidationArtifact, 
#                     data_transformation_config: DataTransformationConfig,):
#         """

#         :param data_validation_artifact: Output reference of data ingestion artifact stage
#         :param data_transformation_config: configuration for data transformation
#         """
#         try:
#             self.data_validation_artifact = data_validation_artifact
#             self.data_transformation_config = data_transformation_config

#         except Exception as e:
#             raise SensorException(e, sys)


#     @staticmethod
#     def read_data(file_path) -> pd.DataFrame:
#         try:
#             return pd.read_csv(file_path)
#         except Exception as e:
#             raise SensorException(e, sys)


#     @classmethod
#     def get_data_transformer_object(cls)->Pipeline:
#         try:
#             robust_scaler = RobustScaler()
#             simple_imputer = SimpleImputer(strategy="constant", fill_value=0)
#             preprocessor = Pipeline(
#                 steps=[
#                     ("Imputer", simple_imputer), #replace missing values with zero
#                     ("RobustScaler", robust_scaler) #keep every feature in same range and handle outlier
#                     ]
#             )
            
#             return preprocessor

#         except Exception as e:
#             raise SensorException(e, sys) from e

    
#     def initiate_data_transformation(self,) -> DataTransformationArtifact:
#         try:
            
#             train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
#             test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)
#             preprocessor = self.get_data_transformer_object()


#             #training dataframe
#             input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
#             target_feature_train_df = train_df[TARGET_COLUMN]
#             target_feature_train_df = target_feature_train_df.replace( TargetValueMapping().to_dict())

#             #testing dataframe
#             input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
#             target_feature_test_df = test_df[TARGET_COLUMN]
#             target_feature_test_df = target_feature_test_df.replace(TargetValueMapping().to_dict())

#             preprocessor_object = preprocessor.fit(input_feature_train_df)
#             transformed_input_train_feature = preprocessor_object.transform(input_feature_train_df)
#             transformed_input_test_feature =preprocessor_object.transform(input_feature_test_df)

#             smt = SMOTETomek(sampling_strategy="minority")

#             input_feature_train_final, target_feature_train_final = smt.fit_resample(
#                 transformed_input_train_feature, target_feature_train_df
#             )

#             input_feature_test_final, target_feature_test_final = smt.fit_resample(
#                 transformed_input_test_feature, target_feature_test_df
#             )

#             train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final) ]
#             test_arr = np.c_[ input_feature_test_final, np.array(target_feature_test_final) ]

#             #save numpy array data
#             save_numpy_array_data( self.data_transformation_config.transformed_train_file_path, array=train_arr, )
#             save_numpy_array_data( self.data_transformation_config.transformed_test_file_path,array=test_arr,)
#             save_object( self.data_transformation_config.transformed_object_file_path, preprocessor_object,)
            
            
#             #preparing artifact
#             data_transformation_artifact = DataTransformationArtifact(
#                 transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
#                 transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
#                 transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
#             )
#             logging.info(f"Data transformation artifact: {data_transformation_artifact}")
#             return data_transformation_artifact
#         except Exception as e:
#             raise SensorException(e, sys) from e


import sys
import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline

from sensor.constant.training_pipeline import TARGET_COLUMN
from sensor.entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact
from sensor.entity.config_entity import DataTransformationConfig
from sensor.exception import SensorException
from sensor.logger import logging
from sensor.ml.model.estimator import TargetValueMapping
from sensor.utils.main_utils import save_numpy_array_data, save_object

class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact, 
                 data_transformation_config: DataTransformationConfig):
        """
        :param data_validation_artifact: Output reference of data ingestion artifact stage
        :param data_transformation_config: configuration for data transformation
        """
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
            logging.info("DataTransformation initialized with validation artifact and transformation config.")
        except Exception as e:
            raise SensorException(f"Error initializing DataTransformation: {str(e)}", sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            logging.info(f"Reading data from file: {file_path}")
            return pd.read_csv(file_path)
        except Exception as e:
            raise SensorException(f"Error reading data from {file_path}: {str(e)}", sys)

    @classmethod
    def get_data_transformer_object(cls) -> Pipeline:
        try:
            robust_scaler = RobustScaler()
            simple_imputer = SimpleImputer(strategy="constant", fill_value=0)
            preprocessor = Pipeline(
                steps=[
                    ("Imputer", simple_imputer),  # Replace missing values with zero
                    ("RobustScaler", robust_scaler)  # Scale features and handle outliers
                ]
            )
            logging.info("Data transformation pipeline created.")
            return preprocessor
        except Exception as e:
            raise SensorException(f"Error creating data transformer object: {str(e)}", sys) from e

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Initiating data transformation process.")

            # Read training and testing data
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)
            preprocessor = self.get_data_transformer_object()

            # Prepare training dataframe
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_train_df = target_feature_train_df.replace(TargetValueMapping().to_dict())
            logging.info(f"Training data shape: {input_feature_train_df.shape}")

            # Prepare testing dataframe
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            target_feature_test_df = target_feature_test_df.replace(TargetValueMapping().to_dict())
            logging.info(f"Testing data shape: {input_feature_test_df.shape}")

            # Fit the preprocessor and transform the data
            preprocessor_object = preprocessor.fit(input_feature_train_df)
            transformed_input_train_feature = preprocessor_object.transform(input_feature_train_df)
            transformed_input_test_feature = preprocessor_object.transform(input_feature_test_df)

            logging.info("Data transformation completed for training and testing features.")

            # Handle class imbalance
            smt = SMOTETomek(sampling_strategy="minority")
            input_feature_train_final, target_feature_train_final = smt.fit_resample(
                transformed_input_train_feature, target_feature_train_df
            )
            input_feature_test_final, target_feature_test_final = smt.fit_resample(
                transformed_input_test_feature, target_feature_test_df
            )

            logging.info("Class imbalance handled using SMOTETomek.")

            # Prepare and save final datasets
            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
            test_arr = np.c_[input_feature_test_final, np.array(target_feature_test_final)]

            # Save numpy array data
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor_object)

            logging.info("Transformed data and preprocessor object saved successfully.")

            # Preparing artifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )
            logging.info(f"Data transformation artifact created: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise SensorException(f"Error during data transformation initiation: {str(e)}", sys) from e
            

