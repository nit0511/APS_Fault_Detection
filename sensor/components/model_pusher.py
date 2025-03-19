
# from sensor.exception import SensorException
# from sensor.logger import logging
# from sensor.entity.artifact_entity import ModelPusherArtifact,ModelTrainerArtifact,ModelEvaluationArtifact
# from sensor.entity.config_entity import ModelEvaluationConfig,ModelPusherConfig
# import os,sys
# from sensor.ml.metric.classification_metric import get_classification_score
# from sensor.utils.main_utils import save_object,load_object,write_yaml_file

# import shutil

# class ModelPusher:

#     def __init__(self,
#                 model_pusher_config:ModelPusherConfig,
#                 model_eval_artifact:ModelEvaluationArtifact):

#         try:
#             self.model_pusher_config = model_pusher_config
#             self.model_eval_artifact = model_eval_artifact
#         except  Exception as e:
#             raise SensorException(e, sys)
    

#     def initiate_model_pusher(self,)->ModelPusherArtifact:
#         try:
#             trained_model_path = self.model_eval_artifact.trained_model_path
            
#             #Creating model pusher dir to save model
#             model_file_path = self.model_pusher_config.model_file_path
#             os.makedirs(os.path.dirname(model_file_path),exist_ok=True)
#             shutil.copy(src=trained_model_path, dst=model_file_path)

#             #saved model dir
#             saved_model_path = self.model_pusher_config.saved_model_path
#             os.makedirs(os.path.dirname(saved_model_path),exist_ok=True)
#             shutil.copy(src=trained_model_path, dst=saved_model_path)

#             #prepare artifact
#             model_pusher_artifact = ModelPusherArtifact(saved_model_path=saved_model_path, model_file_path=model_file_path)
#             return model_pusher_artifact
#         except  Exception as e:
#             raise SensorException(e, sys)


from sensor.exception import SensorException
from sensor.logger import logging
from sensor.entity.artifact_entity import ModelPusherArtifact, ModelEvaluationArtifact
from sensor.entity.config_entity import ModelPusherConfig
import os
import sys
from sensor.utils.main_utils import save_object, load_object, write_yaml_file
import shutil

class ModelPusher:

    def __init__(self,
                 model_pusher_config: ModelPusherConfig,
                 model_eval_artifact: ModelEvaluationArtifact):
        try:
            self.model_pusher_config = model_pusher_config
            self.model_eval_artifact = model_eval_artifact
            logging.info("ModelPusher initialized with configuration and evaluation artifact.")
        except Exception as e:
            raise SensorException(f"Error initializing ModelPusher: {str(e)}", sys)

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        try:
            trained_model_path = self.model_eval_artifact.trained_model_path
            logging.info(f"Trained model path: {trained_model_path}")

            # Creating model pusher dir to save model
            model_file_path = self.model_pusher_config.model_file_path
            os.makedirs(os.path.dirname(model_file_path), exist_ok=True)
            logging.info(f"Model directory created at: {os.path.dirname(model_file_path)}")

            shutil.copy(src=trained_model_path, dst=model_file_path)
            logging.info(f"Copied trained model to: {model_file_path}")

            # Saved model dir
            saved_model_path = self.model_pusher_config.saved_model_path
            os.makedirs(os.path.dirname(saved_model_path), exist_ok=True)
            logging.info(f"Saved model directory created at: {os.path.dirname(saved_model_path)}")

            shutil.copy(src=trained_model_path, dst=saved_model_path)
            logging.info(f"Copied trained model to saved model path: {saved_model_path}")

            # Prepare artifact
            model_pusher_artifact = ModelPusherArtifact(saved_model_path=saved_model_path, model_file_path=model_file_path)
            logging.info(f"Model pusher artifact created: {model_pusher_artifact}")

            return model_pusher_artifact
        except Exception as e:
            raise SensorException(f"Error during model pushing: {str(e)}", sys)
