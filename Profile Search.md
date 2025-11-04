# Building a Large-Scale Profile Search Re-Ranking System at G42

Modern AI platforms rely on search engines for precise, personalized results. At G42, our user-profile search engine had an **average precision of <1%**. To address this, we built a **large-scale ML-based re-ranking system** that increased precision to **87%**, using **AWS SageMaker, EKS, PyTorch, ONNX, Airflow, Kinesis, Terraform, Docker, CI/CD, MLflow, and CloudWatch**. Here’s a full technical walkthrough.

---

## 1. Architecture Overview

Key layers:

1. **Data ingestion & ETL pipelines** – batch & streaming (PySpark, Kinesis).
2. **Feature computation** – scalable extraction for training & real-time inference.
3. **Model training & optimization** – distributed PyTorch + ONNX.
4. **Deployment & orchestration** – Docker, EKS, Terraform, CI/CD.
5. **Monitoring & observability** – MLflow + CloudWatch.

High-level flow:

```
User Query → Feature Pipelines (PySpark + Kinesis) → ML Re-Ranker (PyTorch + ONNX) → Re-Ranked Results → User
```

---

## 2. Scalable ETL & Feature Pipelines

We needed **real-time and batch feature computation**:

* **Batch (PySpark)** – precompute features on historical data.

```python
# PySpark batch pipeline
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DoubleType

spark = SparkSession.builder.appName("FeaturePipeline").getOrCreate()

def compute_similarity(query, profile_text):
    return float(len(set(query.split()) & set(profile_text.split()))) / len(set(query.split()))

similarity_udf = udf(compute_similarity, DoubleType())

profiles_df = spark.read.parquet("s3://g42-data/profiles/")
queries_df = spark.read.parquet("s3://g42-data/search_queries/")

features_df = queries_df.crossJoin(profiles_df)
features_df = features_df.withColumn("similarity_score", similarity_udf(col("query_text"), col("profile_text")))

features_df.write.parquet("s3://g42-features/features.parquet")
```

* **Streaming (Kinesis)** – process updates in real time.

```python
import boto3
import json

# Create Kinesis client
kinesis_client = boto3.client('kinesis')

def send_to_kinesis(profile_update):
    """
    Sends profile updates to Kinesis stream for real-time feature computation
    """
    kinesis_client.put_record(
        StreamName="profile-updates-stream",
        Data=json.dumps(profile_update),
        PartitionKey=str(profile_update["user_id"])
    )

# Example usage
send_to_kinesis({"user_id": 123, "profile_text": "AI engineer with PyTorch experience"})
```

* **Kinesis usage explained**: Kinesis acts as a **real-time streaming buffer**, allowing new profile updates or search queries to be immediately ingested, processed, and passed to the feature pipeline for low-latency inference.

---

## 3. Model Training & ONNX Optimization

We trained a **PyTorch neural network** to predict profile relevance.

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

class ProfileDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

class ReRankerModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc2(self.relu(self.fc1(x))))

dataset = ProfileDataset(features_array, labels_array)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
model = ReRankerModel(input_dim=features_array.shape[1])
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    for x_batch, y_batch in dataloader:
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs.squeeze(), y_batch)
        loss.backward()
        optimizer.step()
```

* **ONNX optimization** for low-latency inference:

```python
dummy_input = torch.randn(1, features_array.shape[1])
torch.onnx.export(model, dummy_input, "re_ranker.onnx", input_names=["input"], output_names=["output"])
```

Result: **Inference latency reduced from ~120ms to <15ms per query**.

---

## 4. Deployment & Infrastructure with Terraform, Docker, and CI/CD

**Terraform** provisions all AWS infrastructure (EKS cluster, S3 buckets, Kinesis streams, CloudWatch log groups):

```hcl
provider "aws" {
  region = "us-east-1"
}

resource "aws_kinesis_stream" "profile_updates" {
  name        = "profile-updates-stream"
  shard_count = 2
}

resource "aws_s3_bucket" "features_bucket" {
  bucket = "g42-features"
}

resource "aws_cloudwatch_log_group" "re_ranker_logs" {
  name = "/g42/re-ranker"
  retention_in_days = 14
}

module "eks" {
  source          = "terraform-aws-modules/eks/aws"
  cluster_name    = "g42-re-ranker-cluster"
  cluster_version = "1.27"
  subnets         = ["subnet-abc", "subnet-def"]
  vpc_id          = "vpc-123"
}
```

* **Docker container** hosts ONNX model:

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
COPY re_ranker.onnx /app/re_ranker.onnx
COPY inference.py /app/inference.py
WORKDIR /app
CMD ["python", "inference.py"]
```

* **Kubernetes Deployment + Service** (same as previous blog).
* **CI/CD**: GitHub Actions triggers build → Docker image → Terraform applies → deploy to EKS.

---

## 5. Monitoring & Observability with MLflow + CloudWatch

* **MLflow** tracks experiments:

```python
import mlflow
mlflow.start_run()
mlflow.log_param("learning_rate", 0.001)
mlflow.log_metric("loss", loss.item())
mlflow.pytorch.log_model(model, "re_ranker_model")
mlflow.end_run()
```

* **CloudWatch** monitors:

```python
import boto3

cloudwatch = boto3.client('cloudwatch')

def log_metrics(latency_ms, precision):
    cloudwatch.put_metric_data(
        Namespace='G42/ReRanker',
        MetricData=[
            {'MetricName': 'InferenceLatency', 'Value': latency_ms, 'Unit': 'Milliseconds'},
            {'MetricName': 'Precision', 'Value': precision, 'Unit': 'Percent'}
        ]
    )

# Example
log_metrics(latency_ms=12.3, precision=87.0)
```

CloudWatch **alerts** if latency spikes >50ms or precision drops <85%, enabling real-time monitoring and alerting.

---

## 6. Results

* **Precision improvement**: <1% → 87%
* **Inference latency**: ~120ms → <15ms per query
* **Scalability**: millions of profiles, thousands of queries/sec
* **Automated retraining**: daily via CI/CD pipelines

---

## 7. Key Learnings

1. **Kinesis** enables real-time streaming updates with low latency.
2. **Terraform** ensures reproducible and compliant infrastructure.
3. **CloudWatch** provides observability with metrics, dashboards, and alerts.
4. **ONNX** drastically reduces inference time.
5. **CI/CD + Docker + EKS** ensures consistent deployments across environments.

This project demonstrates how **cloud-native MLOps, distributed feature engineering, and model optimization** can transform a low-precision search engine into a **highly precise, real-time profile re-ranking system**.

