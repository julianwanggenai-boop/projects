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


----


Below are **two complete training scripts** that integrate with **AWS SageMaker**, **MLflow**, and **ONNX** — one written in **PyTorch**, and one in **TensorFlow**.

Each shows:

* Data loading from S3
* Model definition
* Training and evaluation
* MLflow tracking
* Model saving (and optional ONNX export)

---

## 🔹 `train_pytorch.py`

This script trains a simple churn prediction model in **PyTorch**, designed to run inside SageMaker’s managed training container.

```python
# train_pytorch.py

import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import mlflow
import mlflow.pytorch
import boto3
import io

# --------------------------
# Dataset
# --------------------------
class ChurnDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_parquet(csv_path)
        self.X = torch.tensor(df.drop("churn", axis=1).values, dtype=torch.float32)
        self.y = torch.tensor(df["churn"].values, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# --------------------------
# Model Definition
# --------------------------
class ChurnModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


# --------------------------
# Training Loop
# --------------------------
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = (model(X_batch) > 0.5).float()
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
    return correct / total


# --------------------------
# Main Entry Point
# --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--train", type=str, default="/opt/ml/input/data/train")
    parser.add_argument("--model-dir", type=str, default="/opt/ml/model")
    args = parser.parse_args()

    # MLflow setup
    mlflow.set_tracking_uri("http://mlflow.pepsico.internal")
    mlflow.set_experiment("ChurnPrediction-PyTorch")

    # Load data
    dataset = ChurnDataset(args.train)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChurnModel(input_dim=dataset.X.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Train and evaluate
    with mlflow.start_run():
        for epoch in range(args.epochs):
            loss = train(model, train_loader, criterion, optimizer, device)
            acc = evaluate(model, val_loader, device)
            mlflow.log_metric("loss", loss, step=epoch)
            mlflow.log_metric("accuracy", acc, step=epoch)
            print(f"Epoch {epoch+1}/{args.epochs}, Loss={loss:.4f}, Accuracy={acc:.4f}")

        # Save model to SageMaker model dir
        model_path = os.path.join(args.model_dir, "model.pth")
        torch.save(model.state_dict(), model_path)

        # Log model in MLflow
        mlflow.pytorch.log_model(model, "model")

        # Export to ONNX for optimized inference
        dummy_input = torch.randn(1, dataset.X.shape[1])
        onnx_path = os.path.join(args.model_dir, "model.onnx")
        torch.onnx.export(model, dummy_input, onnx_path, input_names=["input"], output_names=["output"])
```

### How the techs are used:

* **PyTorch** → Defines model and training loop.
* **MLflow** → Logs metrics and saves model versions.
* **ONNX** → Converts PyTorch model for faster inference.
* **SageMaker** → Runs this script on managed GPU/CPU instances automatically.

---

## 🔹 `train_tf.py`

This TensorFlow version trains a similar feed-forward model with Keras and logs to MLflow.

```python
# train_tf.py

import os
import argparse
import pandas as pd
import tensorflow as tf
import mlflow
import mlflow.tensorflow
from sklearn.model_selection import train_test_split

def load_data(path):
    df = pd.read_parquet(path)
    X = df.drop("churn", axis=1).values
    y = df["churn"].values
    return train_test_split(X, y, test_size=0.2, random_state=42)

def build_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation="relu", input_shape=(input_dim,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train", type=str, default="/opt/ml/input/data/train")
    parser.add_argument("--model-dir", type=str, default="/opt/ml/model")
    args = parser.parse_args()

    # MLflow setup
    mlflow.set_tracking_uri("http://mlflow.pepsico.internal")
    mlflow.set_experiment("ChurnPrediction-TensorFlow")

    X_train, X_val, y_train, y_val = load_data(args.train)

    model = build_model(input_dim=X_train.shape[1])

    with mlflow.start_run():
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=2
        )

        # Log metrics
        for epoch, acc in enumerate(history.history["val_accuracy"]):
            mlflow.log_metric("val_accuracy", acc, step=epoch)

        # Save and log model
        model.save(os.path.join(args.model_dir, "tf_model"))
        mlflow.tensorflow.log_model(model, "model")

        # Optionally export to ONNX
        try:
            import tf2onnx
            import onnx
            spec = (tf.TensorSpec((None, X_train.shape[1]), tf.float32, name="input"),)
            model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
            with open(os.path.join(args.model_dir, "model.onnx"), "wb") as f:
                f.write(model_proto.SerializeToString())
        except ImportError:
            print("Skipping ONNX export (install tf2onnx to enable).")
```

### How the techs are used:

* **TensorFlow/Keras** → Builds and trains deep neural networks.
* **MLflow** → Logs performance metrics and model artifacts.
* **ONNX (optional)** → Converts TensorFlow model for cross-platform deployment.
* **SageMaker** → Automatically picks this script as the training entry point.

---

## 🔹 Summary of Integration

| Component                       | Purpose                          | Where It Appears                        |
| ------------------------------- | -------------------------------- | --------------------------------------- |
| **SageMaker**                   | Executes training at scale       | `train_model.py` and script entrypoints |
| **PyTorch / TensorFlow**        | ML frameworks for modeling       | `train_pytorch.py`, `train_tf.py`       |
| **MLflow**                      | Experiment tracking              | Both scripts                            |
| **ONNX**                        | Model optimization for inference | Both scripts                            |
| **S3 (via SageMaker channels)** | Data and model storage           | `--train` and `--model-dir` args        |




