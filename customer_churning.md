# Building a Scalable Customer Churn Prediction MLOps System on AWS

Predicting customer churn is a high-value use case for consumer brands like PepsiCo. A robust MLOps system ensures that such models are not just accurate but also scalable, reproducible, and easy to maintain across teams and production environments.

In this blog, we’ll develop an **end-to-end churn prediction platform** deployed on **AWS SageMaker**, powered by **PyTorch, TensorFlow, Spark, Docker, MLflow, Airflow, and Terraform**.

---

## 1. Architecture Overview

**Core Components:**

| Layer                         | Tech Stack                               | Purpose                                          |
| ----------------------------- | ---------------------------------------- | ------------------------------------------------ |
| Data Ingestion                | AWS S3, Spark, PySpark, Airflow          | Build ETL pipelines for real-time and batch data |
| Model Training                | AWS SageMaker, PyTorch, TensorFlow, ONNX | Train distributed models and optimize inference  |
| Model Tracking                | MLflow                                   | Track experiments, hyperparameters, and versions |
| Containerization & Deployment | Docker, AWS EKS                          | Deploy models as scalable microservices          |
| CI/CD & Infra                 | GitHub Actions, Terraform                | Automate ML pipelines and IaC                    |
| Monitoring                    | AWS CloudWatch                           | Track model performance and drift                |
| Governance                    | Feature Store & Model Registry           | Manage features and model versions               |

---

## 2. Data Pipeline with Spark, PySpark, and Airflow

We’ll create a **data ingestion DAG** in Airflow to pull, clean, and load data into S3 for training.

**Example: `dags/churn_etl_dag.py`**

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from pyspark.sql import SparkSession

def extract_transform_load():
    spark = SparkSession.builder.appName("ChurnETL").getOrCreate()
    
    # Read customer data from multiple sources
    df_customers = spark.read.csv("s3://pepsico-data/customers.csv", header=True, inferSchema=True)
    df_transactions = spark.read.parquet("s3://pepsico-data/transactions/")
    
    # Join and clean
    df = df_customers.join(df_transactions, "customer_id", "left") \
                     .dropna(subset=["age", "tenure", "spend"])
    
    # Feature engineering
    df = df.withColumn("spend_per_month", df["spend"] / df["tenure"])
    
    # Save processed data
    df.write.mode("overwrite").parquet("s3://pepsico-clean/churn_features/")
    spark.stop()

with DAG(
    dag_id='churn_data_pipeline',
    start_date=datetime(2025, 1, 1),
    schedule_interval='@daily'
) as dag:
    etl_task = PythonOperator(
        task_id='etl_process',
        python_callable=extract_transform_load
    )
```

**How techs are used:**

* **Spark/PySpark:** distributed feature computation and joins.
* **Airflow:** orchestrates ETL jobs daily.
* **AWS S3:** data lake for raw and processed data.

---

## 3. Model Training on SageMaker with PyTorch & TensorFlow

We train two models — one in **TensorFlow** and another in **PyTorch** — to compare performance using **MLflow**.

**Example: `train_model.py`**

```python
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.tensorflow import TensorFlow

role = "arn:aws:iam::<account-id>:role/SageMakerExecutionRole"
session = sagemaker.Session()

# PyTorch model training
pytorch_estimator = PyTorch(
    entry_point='train_pytorch.py',
    role=role,
    instance_count=2,
    instance_type='ml.m5.xlarge',
    framework_version='1.13',
    hyperparameters={'epochs': 20, 'lr': 0.001},
    sagemaker_session=session
)
pytorch_estimator.fit({'train': 's3://pepsico-clean/churn_features/'})

# TensorFlow model training
tensorflow_estimator = TensorFlow(
    entry_point='train_tf.py',
    role=role,
    instance_count=2,
    instance_type='ml.m5.xlarge',
    framework_version='2.10',
    hyperparameters={'epochs': 20, 'batch_size': 64},
    sagemaker_session=session
)
tensorflow_estimator.fit({'train': 's3://pepsico-clean/churn_features/'})
```

**How techs are used:**

* **AWS SageMaker:** orchestrates distributed training with managed infrastructure.
* **PyTorch & TensorFlow:** train deep learning models.
* **MLflow:** logs metrics and models for comparison.

---

## 4. Model Optimization with ONNX

Convert the trained model to **ONNX** for faster inference.

```python
import torch
import onnx
from model import ChurnModel

model = ChurnModel()
model.load_state_dict(torch.load("model.pth"))
dummy_input = torch.randn(1, 20)
torch.onnx.export(model, dummy_input, "model.onnx")
onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)
```

**Why:** ONNX allows cross-framework deployment and faster runtime optimizations in SageMaker or EKS.

---

## 5. Containerization and Deployment on AWS EKS

Containerize the ONNX model with **Docker** and deploy it to **EKS** for scalable serving.

**Dockerfile:**

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "serve.py"]
```

**serve.py:**

```python
from fastapi import FastAPI
import onnxruntime as ort
import numpy as np

app = FastAPI()
session = ort.InferenceSession("model.onnx")

@app.post("/predict")
def predict(data: list[float]):
    inputs = np.array(data).astype(np.float32).reshape(1, -1)
    outputs = session.run(None, {"input": inputs})
    return {"churn_probability": float(outputs[0][0])}
```

**Deploying on EKS:**

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

**How techs are used:**

* **Docker:** containerizes the inference service.
* **AWS EKS:** scales containerized model endpoints.
* **ONNX Runtime:** enables optimized inference.

---

## 6. Experiment Tracking with MLflow

During training, log all runs to **MLflow** for reproducibility.

```python
import mlflow
import mlflow.pytorch

mlflow.set_tracking_uri("http://mlflow.pepsico.internal")
mlflow.set_experiment("ChurnPrediction")

with mlflow.start_run():
    model = train()
    metrics = evaluate(model)
    mlflow.log_metrics(metrics)
    mlflow.pytorch.log_model(model, "model")
```

**Benefit:** centralized record of model versions, parameters, and performance metrics.

---

## 7. CI/CD and Infrastructure Automation

Use **GitHub Actions** for CI/CD and **Terraform** for provisioning AWS infrastructure.

**.github/workflows/deploy.yml**

```yaml
name: Deploy MLOps System

on:
  push:
    branches: [main]

jobs:
  build-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build Docker image
        run: docker build -t churn-model:latest .
      - name: Push to ECR
        run: |
          aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
          docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/churn-model:latest
      - name: Deploy via Terraform
        run: |
          terraform init
          terraform apply -auto-approve
```

**Terraform for EKS provisioning:**

```hcl
provider "aws" {
  region = "us-east-1"
}

module "eks" {
  source = "terraform-aws-modules/eks/aws"
  cluster_name = "pepsico-churn-cluster"
  vpc_id = var.vpc_id
  subnet_ids = var.subnet_ids
}
```

**How techs are used:**

* **GitHub Actions:** automates CI/CD for models and infrastructure.
* **Terraform:** provisions AWS EKS, S3, SageMaker roles, etc., as code.

---

## 8. Monitoring and Governance

Integrate **AWS CloudWatch** for model logs and performance metrics, and establish a **feature store** and **model registry**.

```python
import boto3

cloudwatch = boto3.client('cloudwatch')
cloudwatch.put_metric_data(
    Namespace='ChurnModel',
    MetricData=[{
        'MetricName': 'PredictionLatency',
        'Value': latency,
        'Unit': 'Milliseconds'
    }]
)
```

**Feature Store:**

* Store customer features for both training and inference.
* Prevents training-serving skew.

**Model Registry:**

* Keeps approved model versions for deployment.
* Enforces compliance and rollback control.

---

## 9. Results and Impact

* **Automation:** End-to-end ML pipeline from data to deployment.
* **Scalability:** Containerized models autoscale via EKS.
* **Governance:** Reproducible, auditable ML lifecycle with MLflow and registries.
* **Efficiency:** Faster inference through ONNX optimization.
* **Observability:** CloudWatch and Airflow for transparent monitoring.

---

## 10. Summary

This MLOps framework combines the best of modern cloud ML tooling:

* **Data:** Spark + Airflow on S3
* **Training:** SageMaker + PyTorch/TensorFlow
* **Optimization:** ONNX
* **Deployment:** Docker + EKS
* **Automation:** GitHub Actions + Terraform
* **Tracking & Monitoring:** MLflow + CloudWatch

The result is a production-grade, scalable, and automated **Customer Churn Prediction System** that aligns with enterprise MLOps standards for performance, governance, and compliance.

