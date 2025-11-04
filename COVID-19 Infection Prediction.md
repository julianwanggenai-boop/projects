# UAE Government – Individual COVID-19 Infection Prediction Platform: Implementation Deep Dive

To support targeted PCR testing during the COVID-19 pandemic, we developed an **individual-level infection prediction system** on AWS. The platform provides daily risk scores for citizens, enabling authorities to prioritize testing and resources effectively. Here’s how we implemented it.

---

## 1. Architecture Overview

The system was designed with **scalability, real-time prediction, and retraining automation** in mind:

* **Data Ingestion:** Daily updates from hospitals, mobile apps, and testing centers.
* **Feature Engineering:** PySpark pipelines on AWS EMR/S3.
* **Model Training:** Distributed PyTorch on SageMaker, orchestrated with Airflow.
* **Model Deployment:** Containerized inference on AWS EKS with auto-scaling.
* **Monitoring & Logging:** AWS CloudWatch for performance metrics and logging.

The end-to-end workflow is illustrated below:

```
Data Sources → S3 → PySpark ETL → Feature Store → SageMaker Training → Model Registry → Docker Container → EKS Deployment → Daily Risk Scores
```

---

## 2. Data Ingestion and ETL

We used **PySpark** for distributed ETL and **Airflow** for workflow orchestration.

```python
# etl_pipeline.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, to_date

spark = SparkSession.builder.appName("covid_etl").getOrCreate()

# Load raw data from S3
raw_data = spark.read.csv("s3://uae-covid/raw_data/*.csv", header=True, inferSchema=True)

# Preprocess
data = (raw_data
        .withColumn("infection_date", to_date(col("infection_date"), "yyyy-MM-dd"))
        .withColumn("risk_factor", when(col("symptoms") != "", 1).otherwise(0))
        .dropDuplicates(["citizen_id"]))

# Save preprocessed data to S3
data.write.mode("overwrite").parquet("s3://uae-covid/processed_data/")
```

**Airflow DAG to orchestrate ETL:**

```python
# airflow_dag.py
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
from etl_pipeline import run_etl

dag = DAG("covid_etl_dag", start_date=datetime(2025, 1, 1), schedule_interval="@daily")

etl_task = PythonOperator(task_id="run_etl", python_callable=run_etl, dag=dag)
```

---

## 3. Model Training with PyTorch on SageMaker

We built a distributed PyTorch model for predicting individual infection risk.

```python
# train_pytorch.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

# Load preprocessed features
data = pd.read_parquet("s3://uae-covid/processed_data/")
X = torch.tensor(data.drop("infection", axis=1).values, dtype=torch.float32)
y = torch.tensor(data["infection"].values, dtype=torch.float32)

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

class CovidRiskModel(nn.Module):
    def __init__(self, input_dim):
        super(CovidRiskModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.sigmoid(self.fc2(x))

model = CovidRiskModel(X.shape[1])
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    for xb, yb in loader:
        pred = model(xb).squeeze()
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}: loss={loss.item()}")
    
# Save model
torch.save(model.state_dict(), "/opt/ml/model/model.pth")
```

**SageMaker Training Job Command:**

```bash
aws sagemaker create-training-job \
    --training-job-name covid-risk-training \
    --algorithm-specification TrainingImage=763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.1.0-cpu-py39,TrainingInputMode=File \
    --role-arn arn:aws:iam::123456789012:role/SageMakerExecutionRole \
    --input-data-config ChannelName=training,DataSource={S3DataSource={S3DataType=S3Prefix,S3Uri=s3://uae-covid/processed_data/}} \
    --output-data-config S3OutputPath=s3://uae-covid/models/ \
    --resource-config InstanceType=ml.p3.2xlarge,InstanceCount=2,VolumeSizeInGB=50 \
    --stopping-condition MaxRuntimeInSeconds=7200
```

---

## 4. Model Containerization and Deployment on EKS

The trained PyTorch model was containerized with Docker and deployed on **EKS** for **real-time inference**.

**Dockerfile:**

```dockerfile
FROM pytorch/pytorch:2.1.0-cpu
WORKDIR /app
COPY model.pth /app/model.pth
COPY inference.py /app/inference.py
RUN pip install flask pandas
CMD ["python", "inference.py"]
```

**Inference Service (Flask API):**

```python
# inference.py
from flask import Flask, request, jsonify
import torch
from train_pytorch import CovidRiskModel

model = CovidRiskModel(input_dim=10)  # replace with actual input dim
model.load_state_dict(torch.load("model.pth"))
model.eval()

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["features"]
    x = torch.tensor(data, dtype=torch.float32)
    risk_score = model(x).item()
    return jsonify({"risk_score": risk_score})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
```

**Kubernetes Deployment YAML:**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: covid-risk-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: covid-risk
  template:
    metadata:
      labels:
        app: covid-risk
    spec:
      containers:
      - name: covid-risk
        image: 123456789012.dkr.ecr.us-east-1.amazonaws.com/covid-risk:latest
        ports:
        - containerPort: 8080
---
apiVersion: v1
kind: Service
metadata:
  name: covid-risk-service
spec:
  type: LoadBalancer
  selector:
    app: covid-risk
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8080
```

---

## 5. Daily Risk Score Pipeline

The pipeline was scheduled via Airflow:

1. ETL fetches daily updates.
2. Preprocessed features stored in S3.
3. Model inference via EKS service for all citizens.
4. Results saved to DynamoDB/Redshift.
5. High-risk individuals flagged for PCR testing.

---

## 6. Monitoring and Auto-Scaling

* **CloudWatch** monitored API latency and errors.
* **EKS HPA (Horizontal Pod Autoscaler)** scaled pods based on CPU/memory usage.
* **Airflow Logs** tracked ETL and inference pipeline status.

---

## 7. Key Achievements

* Delivered **daily personalized risk scores** for all citizens.
* **Automated retraining** to adapt to evolving pandemic data.
* Achieved **low-latency, high-throughput inference** with containerized PyTorch models on EKS.

---

This implementation demonstrates a **full MLOps lifecycle** using AWS SageMaker, PySpark, Airflow, Docker, and Kubernetes for a high-impact public health application.
