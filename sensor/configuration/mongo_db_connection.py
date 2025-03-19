import pymongo
from sensor.constant.database import DATABASE_NAME
from sensor.constant.env_variable import MONGODB_URL_KEY
import certifi
import os 
from sensor.exception import SensorException
ca = certifi.where()

# DATABASE_NAME = "Sensor_Data"
# COLLECTION_NAME = "APS_sensor"

class MongoDBClient:
    client = None

    def __init__(self, database_name = DATABASE_NAME)-> None:
        try:
            if MongoDBClient.client is None:
                #mongo_db_url = os.getenv(MONGODB_URL_KEY)
                mongo_db_url = "mongodb://localhost:27017"


                if mongo_db_url is None:
                    raise Exception(f"Environment key is not set.")
                MongoDBClient.client = pymongo.MongoClient(mongo_db_url, tls=False)
            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            self.database_name = database_name
        except Exception as e:
            raise e
        