# Building the PAX AI Message Intelligence Platform: From Data to Production AI

In today’s enterprise communication landscape, extracting actionable insights from multi-channel messaging is crucial. The **PAX AI Message Intelligence Platform** was designed to analyze SMS, WhatsApp, and email messages at scale, using state-of-the-art NLP techniques. This blog walks through the end-to-end implementation, covering data pipelines, model training, deployment, monitoring, and infrastructure automation.

---

## 1. Data Ingestion and ETL Pipelines

We built scalable **ETL pipelines** using **PySpark**, **Airflow**, and **AWS Glue** to handle both batch and streaming data from multiple communication sources.

### Airflow DAG for ETL

```python
# airflow_dags/pax_etl_dag.py
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from pyspark.sql import SparkSession

default_args = {
    'owner': 'pax_ai',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG('pax_etl_dag', default_args=default_args, schedule_interval='@daily')

def extract_transform_load():
    spark = SparkSession.builder.appName("PAX_ETL").getOrCreate()
    # Load data from S3
    sms_df = spark.read.json("s3://pax-data/sms/")
    whatsapp_df = spark.read.json("s3://pax-data/whatsapp/")
    email_df = spark.read.json("s3://pax-data/email/")
    
    # Simple ETL transformation example
    all_df = sms_df.unionByName(whatsapp_df).unionByName(email_df)
    all_df = all_df.dropDuplicates(['message_id'])
    
    # Save to feature store (S3)
    all_df.write.mode("overwrite").parquet("s3://pax-feature-store/messages/")

etl_task = PythonOperator(
    task_id='etl_task',
    python_callable=extract_transform_load,
    dag=dag
)
```

**Command to run Airflow locally:**

```bash
# Initialize DB for Airflow
airflow db init

# Start the webserver and scheduler
airflow webserver --port 8080
airflow scheduler
```

### AWS Glue Job

```python
import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job

args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Read raw messages
datasource = glueContext.create_dynamic_frame.from_options(
    connection_type="s3",
    connection_options={"paths": ["s3://pax-data/"]},
    format="json"
)

# Apply transformations
transformed = ApplyMapping.apply(frame=datasource,
    mappings=[("message_id","string","message_id","string"),
              ("text","string","text","string"),
              ("timestamp","string","timestamp","timestamp")])

# Write back to S3 as Parquet
glueContext.write_dynamic_frame.from_options(
    frame=transformed,
    connection_type="s3",
    connection_options={"path": "s3://pax-feature-store/messages/"},
    format="parquet"
)

job.commit()
```

**Run AWS Glue job CLI:**

```bash
aws glue start-job-run --job-name pax-etl-job
```

---

## 2. NLP Model Training with PyTorch

We used **transformer-based models** (BERT, RoBERTa) for entity extraction, event detection, and relationship modeling.

### Example PyTorch Training Script

```python
# train_pytorch.py
import torch
from transformers import BertTokenizer, BertForTokenClassification, Trainer, TrainingArguments
from datasets import load_dataset

# Load dataset
dataset = load_dataset("conll2003")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(batch['tokens'], is_split_into_words=True, truncation=True, padding=True)

dataset = dataset.map(tokenize, batched=True)
model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=dataset['train'].features['ner_tags'].feature.num_classes)

# Training arguments
training_args = TrainingArguments(
    output_dir="./models",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    logging_dir='./logs'
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"]
)

trainer.train()
trainer.save_model("./models/pax_ner_model")
```

**Distributed Training with PyTorch on GPU Cluster:**

```bash
torchrun --nnodes=1 --nproc_per_node=4 train_pytorch.py
```

**ONNX Export for Faster Inference:**

```python
import torch
from transformers import BertForTokenClassification

model = BertForTokenClassification.from_pretrained("./models/pax_ner_model")
dummy_input = torch.randint(0, 1000, (1, 128))
torch.onnx.export(model, dummy_input, "pax_ner_model.onnx", opset_version=12)
```

---

## 3. Deployment with SageMaker & EKS

We containerized models using **Docker**, deployed to **Amazon SageMaker endpoints**, and managed Kubernetes deployments on **EKS**.

### Dockerfile

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY train_pytorch.py .
COPY inference.py .

CMD ["python", "inference.py"]
```

**Build and push Docker image:**

```bash
docker build -t pax-ai-model:latest .
docker tag pax-ai-model:latest <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/pax-ai-model:latest
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com
docker push <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/pax-ai-model:latest
```

### Kubernetes Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pax-ai-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: pax-ai
  template:
    metadata:
      labels:
        app: pax-ai
    spec:
      containers:
      - name: pax-ai-container
        image: <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/pax-ai-model:latest
        ports:
        - containerPort: 8080
```

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: pax-ai-service
spec:
  type: LoadBalancer
  selector:
    app: pax-ai
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8080
```

**Deploy to EKS:**

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl get svc
```

---

## 4. CI/CD Pipeline & MLflow Integration

We automated training and deployment using **GitHub Actions** and **AWS CodePipeline**, with **MLflow** for experiment tracking.

### GitHub Actions Workflow

```yaml
# .github/workflows/pax_ci_cd.yml
name: PAX CI/CD

on:
  push:
    branches: [ main ]

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: 3.10
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Train model
        run: python train_pytorch.py
      - name: Log to MLflow
        run: mlflow run .
      - name: Deploy to EKS
        run: kubectl apply -f k8s/deployment.yaml
```

---

## 5. Monitoring & Logging

We integrated **CloudWatch** and **Prometheus** to monitor system health, model latency, and throughput.

### CloudWatch Logs Example

```bash
# View logs for SageMaker endpoint
aws logs describe-log-groups
aws logs get-log-events --log-group-name "/aws/sagemaker/Endpoints/pax-ai-endpoint"
```

### Prometheus Metrics Example

```yaml
# prometheus.yaml
scrape_configs:
  - job_name: 'pax_ai'
    static_configs:
      - targets: ['pax-ai-service:8080']
```

---

## Conclusion

The **PAX AI Message Intelligence Platform** integrates multiple advanced technologies to deliver real-time insights from multi-source communications. By leveraging **PySpark**, **Airflow**, **AWS Glue**, **PyTorch**, **ONNX**, **SageMaker**, **EKS**, **Docker**, **Terraform**, **CI/CD pipelines**, and **monitoring tools**, the platform achieves scalable, reliable, and high-performance NLP analysis for enterprise messaging data.
ou want me to do that?
