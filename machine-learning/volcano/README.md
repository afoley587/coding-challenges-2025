# ML Model Deployment with Volcano Scheduler

This project demonstrates how to deploy machine learning models on Kubernetes using the Volcano batch scheduler. The setup provides a production-ready deployment that works both on local minikube clusters and production environments.

For a comprehensive guide on deploying ML models with Volcano, see our detailed blog post below.

---

# Deploying ML Models on Kubernetes with Volcano Scheduler: A Production-Ready Guide

Machine learning model deployment in Kubernetes can be challenging, especially when you need reliable scheduling and resource management for GPU workloads. The Volcano scheduler, a batch system built on Kubernetes, provides advanced scheduling capabilities that make it ideal for ML workloads. In this post, we'll walk through deploying a machine learning model using Volcano, with examples that work on both production clusters and local minikube environments.

## What is Volcano?

Volcano is a cloud-native batch scheduling system for Kubernetes, originally developed by Huawei and now part of the Cloud Native Computing Foundation (CNCF) as a sandbox project. Unlike Kubernetes' default scheduler, which is designed primarily for long-running services, Volcano is specifically built to handle batch workloads, high-performance computing (HPC), and machine learning jobs.

At its core, Volcano consists of several key components:

- **Volcano Scheduler**: A high-performance batch scheduler that makes intelligent decisions about pod placement
- **Volcano Controller Manager**: Manages the lifecycle of batch jobs and custom resources
- **Volcano Admission Controller**: Validates and mutates job specifications before they're scheduled

Think of Volcano as a specialized scheduler that understands the unique requirements of compute-intensive workloads, particularly those common in ML and data processing pipelines.

## What Problems Does Volcano Solve?

The default Kubernetes scheduler, while excellent for microservices and web applications, faces several challenges when dealing with batch workloads and ML training jobs:

### 1. **Resource Deadlocks and Starvation**

Consider a scenario where you have a distributed ML training job that needs 4 GPUs across 4 nodes. With the default scheduler:

- Pod 1 might get scheduled and acquire 1 GPU
- Pod 2 might get scheduled and acquire another GPU
- But Pods 3 and 4 can't find available GPUs
- The training job hangs indefinitely, wasting the resources already allocated

This is a classic resource deadlock where partial allocation prevents completion.

### 2. **Inefficient Resource Utilization**

Traditional schedulers often lead to:

- **Resource fragmentation**: Small jobs consuming resources that larger jobs need
- **Priority inversion**: Lower priority jobs blocking higher priority ones
- **Poor GPU utilization**: GPUs sitting idle due to suboptimal scheduling decisions

### 3. **Lack of Job-Level Scheduling**

Kubernetes treats each pod independently, but ML workloads often require:

- **All-or-nothing scheduling**: Either all pods in a job start, or none do
- **Job-aware resource allocation**: Understanding dependencies between pods
- **Coordinated scheduling**: Ensuring pods with communication requirements are placed optimally

### 4. **Fairness and Multi-Tenancy Issues**

In shared clusters:
- One user's large job can monopolize resources
- No fair sharing mechanisms between different teams
- Difficult to implement resource quotas for batch workloads

## How Volcano Solves These Problems

Volcano addresses these challenges through several sophisticated scheduling algorithms and features:

### 1. **Gang Scheduling (All-or-Nothing)**

Volcano's gang scheduling ensures that either all pods in a job get scheduled simultaneously, or none do. This prevents resource deadlocks:

```yaml
spec:
  minAvailable: 4  # All 4 pods must be schedulable before any start
  tasks:
    - replicas: 4
      name: ml-training-workers
```

**How it works:**
- Volcano analyzes the cluster to ensure all required resources are available
- Only when all pods can be scheduled does it actually place them
- If insufficient resources exist, the job waits in a queue rather than partially running

### 2. **Queue-Based Fair Share Scheduling**

Volcano implements sophisticated queue management:

```yaml
apiVersion: scheduling.volcano.sh/v1beta1
kind: Queue
metadata:
  name: ml-team-a
spec:
  weight: 3
  capability:
    cpu: "100"
    memory: "500Gi"
    nvidia.com/gpu: "20"
```

**Benefits:**
- **Fair resource allocation** between teams based on weights
- **Resource quotas** prevent any single team from monopolizing the cluster
- **Priority-based scheduling** ensures critical jobs run first
- **Backfill scheduling** maximizes resource utilization by running smaller jobs when larger ones are queued

### 3. **Advanced Scheduling Algorithms**

Volcano includes multiple scheduling plugins:

#### **Proportion Plugin**
Implements Dominant Resource Fairness (DRF) to ensure fair sharing across multiple resource types (CPU, memory, GPU).

#### **Priority Plugin**
Schedules jobs based on priority levels, with higher priority jobs preempting lower priority ones when necessary.

#### **Gang Plugin**
Ensures all-or-nothing scheduling for jobs that require multiple pods to work together.

#### **Binpack Plugin**
Optimizes resource utilization by packing jobs efficiently to minimize fragmentation.

### 4. **Resource-Aware Scheduling**

Volcano understands modern hardware requirements:

- **GPU topology awareness**: Places pods to optimize GPU communication
- **NUMA alignment**: Considers CPU and memory locality for performance
- **Network bandwidth**: Schedules jobs considering inter-node communication requirements

### 5. **Dynamic Resource Management**

Unlike static scheduling, Volcano provides:

- **Preemption**: Lower priority jobs can be temporarily stopped for urgent work
- **Resource borrowing**: Queues can temporarily use unused resources from other queues
- **Elastic scheduling**: Jobs can scale up/down based on resource availability

## Real-World Example: The ML Training Scenario

Let's see how Volcano handles a real ML training job:

### Without Volcano (Problems):
1. **T+0**: Submit distributed training job requiring 8 GPUs
2. **T+1**: Default scheduler places 3 pods on available nodes
3. **T+5**: Remaining 5 pods can't find GPU resources
4. **T+60**: Job hangs indefinitely, wasting 3 GPUs
5. **Result**: Failed job, wasted resources, frustrated data scientists

### With Volcano (Solution):
1. **T+0**: Submit job to Volcano with `minAvailable: 8`
2. **T+1**: Volcano analyzes cluster - only 6 GPUs available
3. **T+2**: Job queued, no resources wasted
4. **T+15**: Another job completes, freeing 4 GPUs (10 total available)
5. **T+16**: Volcano immediately schedules all 8 pods simultaneously
6. **T+17**: Training begins with all required resources
7. **Result**: Successful job completion, optimal resource utilization

## Why Volcano for ML Workloads?

Machine learning workloads have unique characteristics that make Volcano particularly valuable:

### **Distributed Training Requirements**
- Multi-pod jobs that must start together
- High inter-pod communication bandwidth needs
- Sensitivity to partial failures

### **Resource Intensity**
- Large memory and GPU requirements
- Batch processing patterns
- Variable execution times

### **Multi-Tenancy Needs**
- Multiple teams sharing expensive GPU clusters
- Need for fair resource allocation
- Priority-based scheduling for urgent experiments

### **Cost Optimization**
- Expensive GPU resources must be utilized efficiently
- Minimize idle time through intelligent scheduling
- Prevent resource waste from failed partial allocations

By understanding these requirements and implementing sophisticated scheduling algorithms, Volcano transforms Kubernetes into a platform capable of handling the most demanding ML and HPC workloads efficiently.

## Features

- FastAPI-based ML model serving with health checks
- Volcano scheduler for optimized batch job scheduling
- Gang scheduling to ensure all replicas start together
- Configurable resource management
- Comprehensive monitoring and logging
- Load balancing with multiple replicas
- Automated testing and validation

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Ingress       │    │   Service       │    │   Volcano Job   │
│   (ml-model)    │────│   (ClusterIP)   │────│   (2 replicas)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                               ┌─────────────────┐
                                               │   ML Model Pod  │
                                               │   - FastAPI     │
                                               │   - Scikit-learn│
                                               │   - Health checks│
                                               └─────────────────┘
```

## Prerequisites

- Kubernetes cluster (minikube for local development)
- kubectl configured and connected to your cluster
- Docker for building container images
- Helm 3.x for installing Volcano
- Make utility for running automation scripts

## Quick Start

### For Minikube (Local Development)

```bash
# Complete minikube setup and deployment
make minikube-deploy

# This will:
# 1. Start minikube with appropriate resources
# 2. Install Volcano scheduler
# 3. Build the Docker image
# 4. Deploy the application
# 5. Run tests
```

### For Production Clusters

```bash
# Build and push image (adjust for your registry)
make build
docker tag ml-model:latest your-registry/ml-model:latest
docker push your-registry/ml-model:latest

# Deploy to production
make production-deploy
```

## Manual Deployment Steps

### 1. Install Volcano Scheduler

```bash
make install-volcano
make verify-volcano
```

### 2. Build the Application

```bash
# Build Docker image
make build

# For minikube, build in minikube's Docker environment
make minikube-build
```

### 3. Deploy the Application

```bash
# Deploy Volcano queues and application
make deploy-app
```

### 4. Test the Deployment

```bash
# Run comprehensive tests
make test-deployment

# Run load tests
make load-test

# View logs
make logs
```

## Project Structure

```
├── Makefile              # Automation scripts
├── README.md            # Documentation
├── Dockerfile           # Container definition
├── requirements.txt     # Python dependencies
├── app.py              # FastAPI ML model server
├── test_model.py       # Testing script
└── k8s/
    ├── ml-model-deployment.yaml  # Kubernetes manifests
    └── volcano-queue.yaml        # Volcano queue configuration
```

## API Endpoints

The ML model server exposes the following endpoints:

- `GET /health` - Health check endpoint
- `GET /ready` - Readiness check endpoint
- `GET /model-info` - Model information
- `POST /predict` - Make predictions

### Example Usage

```bash
# Health check
curl http://localhost:8080/health

# Make prediction
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0]]
  }'
```

## Configuration

### Resource Requirements

The default configuration uses minimal resources suitable for minikube:

- CPU Request: 200m, Limit: 1000m
- Memory Request: 512Mi, Limit: 1Gi
- Replicas: 2

### Volcano Configuration

The Volcano job is configured with:

- Gang scheduling with `minAvailable: 2`
- Restart policies for pod failures
- Queue-based scheduling
- Resource quotas

### Environment Variables

- `MODEL_PATH`: Path to the model file (default: `/app/model/model.pkl`)
- `PORT`: Server port (default: `8000`)
- `HOST`: Server host (default: `0.0.0.0`)
- `LOG_LEVEL`: Logging level (default: `info`)

## Monitoring and Debugging

### View Application Status

```bash
make get-status
```

### View Logs

```bash
make logs
```

### Describe Volcano Job

```bash
make describe-job
```

### Access Pod Shell

```bash
make shell
```

### Port Forward for Local Access

```bash
make port-forward
# Service available at http://localhost:8080
```

## Testing

### Automated Testing

```bash
# Run deployment tests
make test-deployment

# Run load tests
make load-test
```

### Manual Testing with Python Script

```bash
# Install requirements for testing
pip install requests

# Run comprehensive test suite
python test_model.py --url http://localhost:8080

# Wait for service to be ready, then test
python test_model.py --url http://localhost:8080 --wait 30
```

## Customization

### Using Your Own Model

1. Replace the sample model creation in `app.py`
2. Update the feature count and model loading logic
3. Rebuild the Docker image
4. Update resource requirements if needed

### Scaling

To adjust the number of replicas:

```bash
# Edit k8s/ml-model-deployment.yaml
# Change replicas: 2 to your desired number
# Also update minAvailable in the Volcano job spec
```

### Resource Adjustment

Update resource requests and limits in the deployment manifest:

```yaml
resources:
  requests:
    cpu: "500m"
    memory: "1Gi"
  limits:
    cpu: "2000m"
    memory: "4Gi"
```

## Production Considerations

### Security

- The container runs as a non-root user
- Health checks prevent unhealthy pods from receiving traffic
- Resource limits prevent resource exhaustion

### High Availability

- Multiple replicas ensure service availability
- Gang scheduling ensures consistent deployments
- Restart policies handle failures gracefully

### Performance

- Configurable resource allocation
- Load balancing across replicas
- Optimized container image with minimal dependencies

### Monitoring

- Health and readiness probes
- Structured logging
- Kubernetes events for troubleshooting

## Cleanup

```bash
# Remove application only
make clean-app

# Remove everything including namespace
make clean-all

# Remove Volcano (optional)
make uninstall-volcano

# Stop minikube
make minikube-stop
```

## Troubleshooting

### Common Issues

1. **Volcano not installed**: Run `make install-volcano`
2. **Image pull errors**: Ensure image is built and available
3. **Resource constraints**: Increase minikube resources or adjust limits
4. **Scheduling issues**: Check Volcano scheduler logs

### Debug Commands

```bash
# Check Volcano installation
kubectl get pods -n volcano-system

# Check application pods
kubectl get pods -n ml-models

# View events
make get-events

# Check Volcano job status
kubectl get vcjob -n ml-models

# View scheduler logs
kubectl logs -l app=volcano-scheduler -n volcano-system
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Support

For issues and questions:

1. Check the troubleshooting section
2. Review Volcano documentation
3. Open an issue in the repository

---

**Note**: This example uses a simple scikit-learn model for demonstration. In production, replace with your actual ML model and adjust resource requirements accordingly.
