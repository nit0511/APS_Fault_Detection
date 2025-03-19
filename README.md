
# APS Fault Detection

## Problem Statement
The Air Pressure System (APS) is a critical component of a heavy-duty vehicle that uses compressed air to force a piston, providing pressure to the brake pads and slowing the vehicle down. The advantages of using an APS over a hydraulic system include the easy availability and long-term sustainability of natural air.

This project addresses a binary classification problem, where the affirmative class indicates that a failure was caused by a specific component of the APS, while the negative class indicates that the failure was caused by components unrelated to the APS system.

## Proposed Solution
The focus of this project is on the APS, which generates pressurized air for various functions in a truck, including braking and gear changes. The dataset's positive class corresponds to component failures within the APS, while the negative class relates to failures in other truck components.

The main goal is to minimize unnecessary repairs and reduce the cost associated with false predictions.

## Tech Stack Used
- Python
- FastAPI
- Machine Learning Algorithms
- Docker
- MongoDB

## Infrastructure Required
- AWS S3
- AWS EC2
- AWS ECR
- GitHub Actions
- Terraform

## How to Run

### Prerequisites
- Ensure MongoDB is installed locally, along with MongoDB Compass for data storage.
- Create an AWS account to access services like S3, ECR, and EC2 instances.

### Step-by-Step Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/nit0511/Sensor-Fault-Detection.git
   ```
2. **Create a Conda Environment**
   ```bash
   conda create -n sensor python=3.8 -y
   conda activate sensor
   ```
3. **Install the Requirements**
   ```bash
   pip install -r requirements.txt
   ```
4. **Export Environment Variables**
   ```bash
   export AWS_ACCESS_KEY_ID=<AWS_ACCESS_KEY_ID>
   export AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY>
   export AWS_DEFAULT_REGION=<AWS_DEFAULT_REGION>
   export MONGODB_URL="mongodb+srv://<username>:<password>@your_mongodb_url"
   ```
5. **Run the Application Server**
   ```bash
   python main.py
   ```
6. **Train the Application** Access the training endpoint at:
   ```bash
   http://localhost:8080/train
   ```
7. **Prediction Application** Access the prediction endpoint at:
   ```bash
   http://localhost:8080/predict
   ```

### Running Locally with Docker

1. **Check for Dockerfile** Ensure the Dockerfile is available in the project directory.
2. **Build the Docker Image**
   ```bash
     docker build --build-arg AWS_ACCESS_KEY_ID=<AWS_ACCESS_KEY_ID> \
    --build-arg AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY> \
    --build-arg AWS_DEFAULT_REGION=<AWS_DEFAULT_REGION> \
    --build-arg MONGODB_URL=<MONGODB_URL> .
   ```
3. **Run the Docker Image**
   ```bash
   docker run -d -p 8080:8080 <IMAGE_NAME>
   ```
### MongoDB URL Example
For Windows users:
  ```bash
MONGO_DB_URL=mongodb+srv://<username>:<password>@your_mongodb_url
  ```
For Linux users:
  ```bash
MONGO_DB_URL=mongodb+srv://<username>:<password>@your_mongodb_url
  ```
Then run:
  ```bash
python main.py
  ```
