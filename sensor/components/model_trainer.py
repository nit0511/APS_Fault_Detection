# from sensor.utils.main_utils import load_numpy_array_data
# from sensor.exception import SensorException
# from sensor.logger import logging
# from sensor.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
# from sensor.entity.config_entity import ModelTrainerConfig
# import os,sys
# from xgboost import XGBClassifier
# from sensor.ml.metric.classification_metric import get_classification_score
# from sensor.ml.model.estimator import SensorModel
# from sensor.utils.main_utils import save_object,load_object
# class ModelTrainer:

#     def __init__(self,model_trainer_config:ModelTrainerConfig,
#         data_transformation_artifact:DataTransformationArtifact):
#         try:
#             self.model_trainer_config=model_trainer_config
#             self.data_transformation_artifact=data_transformation_artifact
#         except Exception as e:
#             raise SensorException(e,sys)

#     def perform_hyper_paramter_tunig(self):...
    

#     def train_model(self,x_train,y_train):
#         try:
#             xgb_clf = XGBClassifier()
#             xgb_clf.fit(x_train,y_train)
#             return xgb_clf
#         except Exception as e:
#             raise e
    
#     def initiate_model_trainer(self)->ModelTrainerArtifact:
#         try:
#             train_file_path = self.data_transformation_artifact.transformed_train_file_path
#             test_file_path = self.data_transformation_artifact.transformed_test_file_path

#             #loading training array and testing array
#             train_arr = load_numpy_array_data(train_file_path)
#             test_arr = load_numpy_array_data(test_file_path)

#             x_train, y_train, x_test, y_test = (
#                 train_arr[:, :-1],
#                 train_arr[:, -1],
#                 test_arr[:, :-1],
#                 test_arr[:, -1],
#             )

#             model = self.train_model(x_train, y_train)
#             y_train_pred = model.predict(x_train)
#             classification_train_metric =  get_classification_score(y_true=y_train, y_pred=y_train_pred)
            
#             if classification_train_metric.f1_score<=self.model_trainer_config.expected_accuracy:
#                 raise Exception("Trained model is not good to provide expected accuracy")
            
#             y_test_pred = model.predict(x_test)
#             classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)


#             #Overfitting and Underfitting
#             diff = abs(classification_train_metric.f1_score-classification_test_metric.f1_score)
            
#             if diff>self.model_trainer_config.overfitting_underfitting_threshold:
#                 raise Exception("Model is not good try to do more experimentation.")

#             preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            
#             model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
#             os.makedirs(model_dir_path,exist_ok=True)
#             sensor_model = SensorModel(preprocessor=preprocessor,model=model)
#             save_object(self.model_trainer_config.trained_model_file_path, obj=sensor_model)

#             #model trainer artifact

#             model_trainer_artifact = ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path, 
#             train_metric_artifact=classification_train_metric,
#             test_metric_artifact=classification_test_metric)
#             logging.info(f"Model trainer artifact: {model_trainer_artifact}")
#             return model_trainer_artifact
#         except Exception as e:
#             raise SensorException(e,sys)


from sensor.utils.main_utils import load_numpy_array_data
from sensor.exception import SensorException
from sensor.logger import logging
from sensor.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from sensor.entity.config_entity import ModelTrainerConfig
import os
import sys
from xgboost import XGBClassifier
from sensor.ml.metric.classification_metric import get_classification_score
from sensor.ml.model.estimator import SensorModel
from sensor.utils.main_utils import save_object, load_object

class ModelTrainer:

    def __init__(self, model_trainer_config: ModelTrainerConfig,
                 data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
            logging.info("ModelTrainer initialized with configuration and transformation artifact.")
        except Exception as e:
            raise SensorException(f"Error initializing ModelTrainer: {str(e)}", sys)

    def perform_hyper_parameter_tuning(self):
        # Implementation for hyperparameter tuning would go here
        logging.info("Hyperparameter tuning started.")
        pass

    def train_model(self, x_train, y_train):
        try:
            logging.info("Training the XGBoost model.")
            xgb_clf = XGBClassifier()
            xgb_clf.fit(x_train, y_train)
            logging.info("Model training completed successfully.")
            return xgb_clf
        except Exception as e:
            raise SensorException(f"Error during model training: {str(e)}", sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            logging.info(f"Loading training data from: {train_file_path}")
            train_arr = load_numpy_array_data(train_file_path)
            logging.info("Training data loaded successfully.")

            logging.info(f"Loading testing data from: {test_file_path}")
            test_arr = load_numpy_array_data(test_file_path)
            logging.info("Testing data loaded successfully.")

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            logging.info("Starting model training process.")
            model = self.train_model(x_train, y_train)

            y_train_pred = model.predict(x_train)
            classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)
            logging.info(f"Training metrics: {classification_train_metric}")

            if classification_train_metric.f1_score <= self.model_trainer_config.expected_accuracy:
                logging.warning("Trained model is not good enough to provide expected accuracy.")
                raise SensorException("Trained model is not good enough to provide expected accuracy.")

            y_test_pred = model.predict(x_test)
            classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)
            logging.info(f"Testing metrics: {classification_test_metric}")

            # Overfitting and Underfitting
            diff = abs(classification_train_metric.f1_score - classification_test_metric.f1_score)
            logging.info(f"Difference between train and test F1 scores: {diff}")

            if diff > self.model_trainer_config.overfitting_underfitting_threshold:
                logging.warning("Model is not performing well; consider further experimentation.")
                raise SensorException("Model is not performing well; consider further experimentation.")

            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            logging.info("Preprocessor loaded successfully.")

            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path, exist_ok=True)
            logging.info(f"Model directory created at: {model_dir_path}")

            sensor_model = SensorModel(preprocessor=preprocessor, model=model)
            save_object(self.model_trainer_config.trained_model_file_path, obj=sensor_model)
            logging.info(f"Trained model saved at: {self.model_trainer_config.trained_model_file_path}")

            # Model trainer artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=classification_train_metric,
                test_metric_artifact=classification_test_metric
            )
            logging.info(f"Model trainer artifact created: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise SensorException(f"Error during model training initiation: {str(e)}", sys)
