# Building a Large-Scale Profile Search Re-Ranking System at G42

Modern AI-driven platforms increasingly rely on search engines to deliver precise and personalized results. At G42, we faced a challenge: our user-profile search engine had an **average precision of less than 1%**. To tackle this, we designed and implemented a **large-scale machine learning–based re-ranking system** that boosted precision to **87%**, leveraging a combination of AWS SageMaker, PyTorch, ONNX, Airflow, and CI/CD automation. Here’s a detailed walkthrough of the project, covering each component, implementation action, and outcome.

---

## 1. Architecture Overview

Our system is composed of several key layers:

1. **Data ingestion and ETL pipelines** – to process raw user-profile data in batch and streaming modes.
2. **Feature computation** – scalable feature extraction using PySpark and Kinesis.
3. **Model training and optimization** – distributed PyTorch models with ONNX export for low-latency inference.
4. **Deployment and orchestration** – via Docker, EKS, Terraform, and CI/CD pipelines.
5. **Monitoring and observability** – using MLflow and CloudWatch.

The high-level flow looks like this:

```
User Search Query → Feature Pipelines (PySpark + Kinesis) → ML Re-Ranker (PyTorch + ONNX) → Re-Ranked Results → User
```

---

## 2. Scalable ETL and Feature Pipelines

We needed **dynamic, high-throughput feature computation** for millions of profiles:

* **Batch pipeline**: PySpark jobs running on EMR to preprocess historical data.
* **Streaming pipeline**: AWS Kinesis streams to capture real-time profile updates.

Example PySpark job snippet for feature extraction:

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DoubleType

spark = SparkSession.builder.appName("FeaturePipeline").getOrCreate()

# Example feature: similarity score between search query and profile description
def compute_similarity(query, profile_text):
    return float(len(set(query.split()) & set(profile_text.split())))/len(set(query.split()))

similarity_udf = udf(compute_similarity, DoubleType())

profiles_df = spark.read.parquet("s3://g42-data/profiles/")
queries_df = spark.read.parquet("s3://g42-data/search_queries/")

features_df = queries_df.crossJoin(profiles_df)
features_df = features_df.withColumn("similarity_score", similarity_udf(col("query_text"), col("profile_text")))

features_df.write.parquet("s3://g42-features/features.parquet")
```

* **Streaming update** via Kinesis:

```python
import boto3
import json

kinesis_client = boto3.client('kinesis')

def send_to_stream(profile_update):
    kinesis_client.put_record(
        StreamName="profile-updates",
        Data=json.dumps(profile_update),
        PartitionKey=profile_update["user_id"]
    )
```

These pipelines fed features for both training and inference in real-time.

---

## 3. Model Training and ONNX Optimization

The core of the re-ranking engine is a **PyTorch-based neural network** trained to predict the relevance of profiles given a query.

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

# Load data
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

To reduce inference latency, we exported the PyTorch model to **ONNX**:

```python
dummy_input = torch.randn(1, features_array.shape[1])
torch.onnx.export(model, dummy_input, "re_ranker.onnx", input_names=["input"], output_names=["output"])
```

**Result**: ONNX inference latency dropped from ~120ms per query to <15ms per query.

---

## 4. Deployment with EKS, Docker, and CI/CD

We containerized the re-ranker using Docker:

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
COPY re_ranker.onnx /app/re_ranker.onnx
COPY inference.py /app/inference.py
WORKDIR /app
CMD ["python", "inference.py"]
```

**Kubernetes deployment (`k8s/deployment.yaml`)**:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: re-ranker
spec:
  replicas: 3
  selector:
    matchLabels:
      app: re-ranker
  template:
    metadata:
      labels:
        app: re-ranker
    spec:
      containers:
      - name: re-ranker
        image: g42/re-ranker:latest
        ports:
        - containerPort: 8080
```

**Service (`k8s/service.yaml`)**:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: re-ranker-service
spec:
  selector:
    app: re-ranker
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8080
  type: LoadBalancer
```

We automated deployment using **Terraform** for infrastructure provisioning and **GitHub Actions + CodePipeline** for CI/CD.

---

## 5. Model Observability and Monitoring

* **MLflow** tracked experiments, model versions, and metrics.
* **AWS CloudWatch** visualized key metrics like inference latency and feature distribution drift.

Example MLflow logging:

```python
import mlflow
mlflow.start_run()
mlflow.log_param("lr", 0.001)
mlflow.log_metric("loss", loss.item())
mlflow.pytorch.log_model(model, "re_ranker_model")
mlflow.end_run()
```

---

## 6. Results

* **Precision improvement**: <1% → 87%
* **Inference latency**: ~120ms → <15ms per query
* **Scalability**: Handled millions of profiles and thousands of queries per second
* **Automated retraining**: New model versions deployed daily via CI/CD

---

## 7. Lessons Learned

1. **ONNX optimization** drastically reduces latency in large-scale inference.
2. **CI/CD for MLOps** ensures reproducibility and governance across multiple environments.
3. **Streaming + batch pipelines** allow real-time responsiveness without sacrificing historical insights.
4. **Close collaboration with data engineering teams** is crucial for designing secure and compliant high-throughput pipelines.

---

This project demonstrates how a combination of **cloud-native MLOps, distributed feature engineering, and model optimization** can transform a low-precision search engine into a **highly precise, real-time profile re-ranking system**.


