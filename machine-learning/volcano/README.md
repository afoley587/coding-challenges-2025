# Deploying ML Models on Kubernetes with Volcano Scheduler: A Production-Ready Guide

Machine learning model deployment in Kubernetes can be challenging,
especially when you need reliable scheduling and resource management for
GPU workloads.
The Volcano scheduler, a batch system built on Kubernetes, provides
advanced scheduling capabilities that make it ideal for ML workloads.
In this post, we'll walk through deploying a machine learning model using
Volcano, with examples that work on both production clusters and local
minikube environments.

Please note that this blog post will focus more on
scheduling the ML workloads in kubernetes and not any
model building or generation.
For the entire codebase,
please visit my
[GitHub repo](https://github.com/afoley587/coding-challenges-2025/tree/main/machine-learning/volcano)!

## What is Volcano?

Volcano is a cloud-native batch scheduling system for Kubernetes,
originally developed by Huawei and now part of the Cloud Native Computing
Foundation (CNCF) as a sandbox project.
Unlike Kubernetes' default scheduler, which is designed primarily for
long-running services, Volcano is specifically built to handle batch
workloads, high-performance computing (HPC), and machine learning jobs.

At its core, Volcano consists of several key components:

- Volcano Scheduler: A high-performance batch scheduler that makes
  intelligent decisions about pod placement
- Volcano Controller Manager: Manages the lifecycle of batch jobs and
  custom resources
- Volcano Admission Controller: Validates and mutates job specifications
  before they're scheduled

Think of Volcano as a specialized scheduler that understands the unique
requirements of compute-intensive workloads, particularly those common in
ML and data processing pipelines.

## What Problems Does Volcano Solve?

The default Kubernetes scheduler, while excellent for microservices and
web applications, faces several challenges when dealing with batch
workloads and ML training jobs:

### 1. Resource Deadlocks and Starvation

Consider a scenario where you have a distributed ML training job that
needs 4 GPUs across 4 nodes.
With the default scheduler:

- Pod 1 might get scheduled and acquire 1 GPU
- Pod 2 might get scheduled and acquire another GPU
- But Pods 3 and 4 can't find available GPUs
- The training job hangs indefinitely, wasting the resources already
  allocated

This can create a resource deadlock where partial allocation prevents
completion.

### 2. Inefficient Resource Utilization

Traditional schedulers may lead to:

- Resource fragmentation: Small jobs consuming resources that larger jobs
  need
- Priority inversion: Lower priority jobs blocking higher priority ones
- Poor GPU utilization: GPUs sitting idle due to suboptimal scheduling
  decisions

### 3. Lack of Job-Level Scheduling

Kubernetes treats each pod independently, but ML workloads often require:

- All-or-nothing scheduling: Either all pods in a job start, or none do
- Job-aware resource allocation: Understanding dependencies between pods
- Coordinated scheduling: Ensuring pods with communication requirements
  are placed optimally

### 4. Fairness and Multi-Tenancy Issues

In shared clusters:
- One user's large job can monopolize resources
- No fair sharing mechanisms between different teams
- Difficult to implement resource quotas for batch workloads

## How Volcano Solves These Problems

Volcano addresses these challenges through several sophisticated
scheduling algorithms and features:

### 1. Gang Scheduling (All-or-Nothing)

Volcano's gang scheduling ensures that either all pods in a job get
scheduled simultaneously, or none do.
This prevents resource deadlocks:

```yaml
spec:
  minAvailable: 4  # All 4 pods must be schedulable before any start
  tasks:
    - replicas: 4
      name: ml-training-workers
```

How it works:
- Volcano analyzes the cluster to ensure all required resources are
  available
- Only when all pods can be scheduled does it actually place them
- If insufficient resources exist, the job waits in a queue rather than
  partially running

### 2. Queue-Based Fair Share Scheduling

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

Benefits:
- Fair resource allocation between teams based on weights
- Resource quotas prevent any single team from monopolizing the cluster
- Priority-based scheduling ensures critical jobs run first
- Backfill scheduling maximizes resource utilization by running smaller
  jobs when larger ones are queued

### 3. Advanced Scheduling Algorithms

Volcano includes multiple scheduling plugins:

#### Proportion Plugin
Implements Dominant Resource Fairness (DRF) to ensure fair sharing across
multiple resource types (CPU, memory, GPU).

#### Priority Plugin
Schedules jobs based on priority levels, with higher priority jobs
preempting lower priority ones when necessary.

#### Gang Plugin
Ensures all-or-nothing scheduling for jobs that require multiple pods to
work together.

#### Binpack Plugin
Optimizes resource utilization by packing jobs efficiently to minimize
fragmentation.

### 4. Resource-Aware Scheduling

Volcano understands modern hardware requirements:

- GPU topology awareness: Places pods to optimize GPU communication
- NUMA alignment: Considers CPU and memory locality for performance
- Network bandwidth: Schedules jobs considering inter-node communication
  requirements

### 5. Dynamic Resource Management

Unlike static scheduling, Volcano provides:

- Preemption: Lower priority jobs can be temporarily stopped for urgent
  work
- Resource borrowing: Queues can temporarily use unused resources from
  other queues
- Elastic scheduling: Jobs can scale up/down based on resource availability

## Real-World Example: The ML Training Scenario

Let's see how Volcano handles a real ML training job:

### Without Volcano (Problems):
1. T+0: Submit distributed training job requiring 8 GPUs
2. T+1: Default scheduler places 3 pods on available nodes
3. T+5: Remaining 5 pods can't find GPU resources
4. T+60: Job hangs indefinitely, wasting 3 GPUs
5. Result: Failed job, wasted resources, frustrated data scientists

### With Volcano (Solution):
1. T+0: Submit job to Volcano with `minAvailable: 8`
2. T+1: Volcano analyzes cluster - only 6 GPUs available
3. T+2: Job queued, no resources wasted
4. T+15: Another job completes, freeing 4 GPUs (10 total available)
5. T+16: Volcano immediately schedules all 8 pods simultaneously
6. T+17: Training begins with all required resources
7. Result: Successful job completion, optimal resource utilization

## Why Volcano for ML Workloads?

Machine learning workloads have unique characteristics that make Volcano
particularly valuable:

### Distributed Training Requirements
- Multi-pod jobs that must start together
- High inter-pod communication bandwidth needs
- Sensitivity to partial failures

### Resource Intensity
- Large memory and GPU requirements
- Batch processing patterns
- Variable execution times

### Multi-Tenancy Needs
- Multiple teams sharing expensive GPU clusters
- Need for fair resource allocation
- Priority-based scheduling for urgent experiments

### Cost Optimization
- Expensive GPU resources must be utilized efficiently
- Minimize idle time through intelligent scheduling
- Prevent resource waste from failed partial allocations

By understanding these requirements and implementing sophisticated
scheduling algorithms, Volcano transforms Kubernetes into a platform
capable of handling the most demanding ML and HPC workloads efficiently.

## Features

- FastAPI-based ML model serving with health checks
- Volcano scheduler for optimized batch job scheduling
- Gang scheduling to ensure all replicas start together
- Configurable resource management
- Comprehensive monitoring and logging
- Load balancing with multiple replicas
- Automated testing and validation

## Prerequisites

- Kubernetes cluster (minikube for local development)
- kubectl configured and connected to your cluster
- Docker for building container images
- Helm 3.x for installing Volcano
- Make utility for running automation scripts

## Installing Volcano and Deploying the Application

Understanding each step of the deployment process is important for Site
Reliability Engineers managing ML infrastructure.
Below I will try to break down our installation of Volcano and the
deployment of our model.

### 0. Start a Local Minikube Cluster

We are going to be running this entire stack locally via `minikube`.
You can start the minikube cluster with:

```bash
make minikube-start

# Or manually via
# minikube start --cpus=4 --memory=4096 --disk-size=20g
# minikube addons enable ingress
```

### 1. Install Volcano Scheduler

Next, we need to install the Volcano components and verify they're
properly installed:

```bash
make install-volcano
make verify-volcano

# Or manually via
# helm repo add volcano-sh https://volcano-sh.github.io/helm-charts
# helm repo update
# kubectl create namespace volcano-system --dry-run=client -o yaml | kubectl apply -f -
# helm upgrade --install $(HELM_RELEASE_NAME) volcano-sh/volcano \
#   --namespace volcano-system \
#   --version v1.12.0 \
#   --set basic.image.tag=v1.12.0 \
#   --wait --timeout=300s
# kubectl wait --for=condition=ready pod -l app=volcano-scheduler --namespace volcano-system --timeout=300s
# kubectl get pods -n volcano-system
# kubectl get crd | grep volcano
```

What this does:
Volcano installation involves deploying several control-plane components
to your Kubernetes cluster:

- Volcano Scheduler: Replaces the default Kubernetes scheduler for batch
  workloads.
  This component implements advanced scheduling algorithms like gang
  scheduling, fair-share scheduling, and resource-aware placement.

- Volcano Controller Manager: Manages the lifecycle of Volcano-specific
  custom resources (VCJobs, Queues, PodGroups).
  It watches for job submissions and coordinates with the scheduler to
  ensure proper resource allocation.

- Volcano Admission Controller: Acts as an entry point that validates and
  mutates incoming job specifications.
  It ensures jobs meet resource requirements and policy constraints before
  they're submitted to the scheduler.

Why it's needed:
The default Kubernetes scheduler is optimized for long-running services
and microservices.
It lacks the sophisticated resource management needed for ML workloads,
particularly:
- Gang scheduling for distributed training jobs
- Fair resource sharing between multiple teams
- Advanced preemption and priority handling
- GPU topology awareness for optimal placement

What gets installed:

- volcano-scheduler (deployment)
- volcano-controller (deployment)
- volcano-admission (deployment)
- Custom Resource Definitions (VCJob, Queue, PodGroup)
- RBAC permissions and service accounts
- Webhook configurations for admission control

```bash
$ kubectl get pod -n volcano-system
NAME                                          READY   STATUS              RESTARTS   AGE
volcano-system-admission-55d8648fd7-46brq     1/1     Running   0          51s
volcano-system-controllers-86455cb889-vjlw6   1/1     Running   0          51s
volcano-system-scheduler-768f8d999b-lgrkj     1/1     Running   0          51s
```

### 2. Build the Application

Now, we will build our python application:

```bash
# For minikube - build image locally
make minikube-build

# Or manually via
# eval $(minikube docker-env) && docker build -t ml-model:latest ./python/
```

What the build process does:

This step builds and packages our application with docker.
It does so within the minikube context so minikube can see and deploy the
image.
We will omit any in-depth discussion about this step.

### 3. Deploy the Application

Next, we can deploy our python application and the
Volcano resources:

```bash
make deploy-app

# Or manually via
# kubectl create namespace ml-models --dry-run=client -o yaml | kubectl apply -f -
# kubectl apply -f k8s/volcano-queue.yaml
# kubectl apply -f k8s/ml-model-deployment.yaml
```

#### What gets deployed:

This step creates the ML serving infrastructure with multiple Kubernetes
resources:

##### Volcano Queue Configuration:

```yaml
apiVersion: scheduling.volcano.sh/v1beta1
kind: Queue
metadata:
  name: ml-queue
  # Queues are cluster-wide, so no namespace needed
spec:
  weight: 1
  reclaimable: true
  capability:
    cpu: "4"
    memory: "8Gi"

---
apiVersion: scheduling.volcano.sh/v1beta1
kind: Queue
metadata:
  name: default
  # Queues are cluster-wide, so no namespace needed
spec:
  weight: 1
  reclaimable: true
```

What queues provide:

- Resource Management: Each queue has defined resource limits (CPU, memory,
  GPU)
- Fair Sharing: Weight-based allocation ensures fair resource distribution
  between teams
- Resource Reclamation: reclaimable: true allows unused resources to be
  borrowed by other queues
- Priority Scheduling: Jobs are scheduled based on queue priority and
  available resources
- Multi-tenancy: Different teams can use different queues with isolated
  resource quotas

Why queues matter for ML workloads:
In a shared cluster, queues prevent scenarios like:

- One team's large training job consuming all cluster resources
- Critical inference services being starved by batch training jobs
- Resource contention between different ML workflows

##### Configuration Management:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ml-model-config
data:
  MODEL_PATH: "/app/model/model.pkl"
  PORT: "8000"
  # ... other config
```

Centralizes application configuration, making it easy to modify settings
without rebuilding images.

##### Volcano Job Definition:

```yaml
apiVersion: batch.volcano.sh/v1alpha1
kind: Job
metadata:
  name: ml-model-job
  namespace: ml-models
spec:
  # Gang scheduling - ensures all 2 pods start together or none start
  minAvailable: 2
  # Use Volcano scheduler instead of default Kubernetes scheduler
  schedulerName: volcano
  # Define policies for handling pod failures and evictions
  policies:
    - event: PodEvicted    # If pod gets evicted, restart entire job
      action: RestartJob
    - event: PodFailed     # If pod fails, restart entire job
      action: RestartJob
  # Volcano plugins configuration (empty arrays use defaults)
  plugins:
    svc: []    # Service plugin for inter-pod communication
    ssh: []    # SSH plugin for secure pod-to-pod access
  # Maximum number of job restart attempts before giving up
  maxRetry: 3
  # Assign this job to the "default" queue for resource management
  queue: default
  # Define the tasks that make up this job
  tasks:
    # Single task type with 2 replicas for high availability
    - replicas: 2
      name: ml-server
      template:
        metadata:
          labels:
            app: ml-model
            role: server
        spec:
          restartPolicy: OnFailure
          containers:
            - name: ml-model
              image: ml-model:latest
              imagePullPolicy: IfNotPresent
              ports:
                - containerPort: 8000
                  name: http
              envFrom:
                - configMapRef:
                    name: ml-model-config
              resources:
                requests:         # Guaranteed resources
                  cpu: "200m"
                  memory: "512Mi"
                limits:           # Maximum allowed resources
                  cpu: "1000m"
                  memory: "1Gi"
              livenessProbe:
                httpGet:
                  path: /health
                  port: 8000
                initialDelaySeconds: 30
                periodSeconds: 10
                timeoutSeconds: 5
                failureThreshold: 3
              readinessProbe:
                httpGet:
                  path: /ready
                  port: 8000
                initialDelaySeconds: 10
                periodSeconds: 5
                timeoutSeconds: 3
                failureThreshold: 3
              volumeMounts:
                - name: model-storage
                  mountPath: /app/model
          volumes:
            - name: model-storage
              emptyDir: {}
```

Key Volcano scheduling features enabled:
- Gang Scheduling: Ensures all replicas start simultaneously, preventing
  resource deadlocks
- Job-level Policies: Defines how to handle pod failures and evictions
- Resource Coordination: Guarantees all required resources before starting
  any pods
- Queue Integration: Enables fair resource sharing with other workloads

The default configuration uses minimal resources suitable for minikube:

- CPU Request: 200m, Limit: 1000m
- Memory Request: 512Mi, Limit: 1Gi
- Replicas: 2

The Volcano job is configured with:

- Gang scheduling with `minAvailable: 2`
- Restart policies for pod failures
- Queue-based scheduling
- Resource quotas

##### Service and Ingress:

```yaml
# ClusterIP Service for internal load balancing
apiVersion: v1
kind: Service
spec:
  selector:
    app: ml-model
  ports:
    - port: 80
      targetPort: 8000

# Ingress for external access
apiVersion: networking.k8s.io/v1
kind: Ingress
spec:
  rules:
    - host: ml-model.local
```

Allows ingress so that we can interact with our ML model
(via `curl`, `wget`, etc.).

### 4. Test the Deployment

#### Infrastructure Health Checks:

We can verify all pods and services are deployed as we expect.
We should see two pods running which should match the `replicas`
of our Volcano job.

```bash
$ kubectl get pods -n ml-models -o wide
NAME                       READY   STATUS    RESTARTS   AGE   IP            NODE       NOMINATED NODE   READINESS GATES
ml-model-job-ml-server-0   1/1     Running   0          55s   10.244.0.11   minikube   <none>           <none>
ml-model-job-ml-server-1   1/1     Running   0          55s   10.244.0.10   minikube   <none>           <none>

$ kubectl get vcjob -n ml-models -o wide
NAME           STATUS    MINAVAILABLE   RUNNINGS   AGE   QUEUE
ml-model-job   Running   2              2          57s   ml-queue
```

#### API Endpoint Validation:

The ML model server exposes the following endpoints:

- `GET /health` - Health check endpoint
- `GET /ready` - Readiness check endpoint
- `GET /model-info` - Model information
- `POST /predict` - Make predictions

First, `port-forward` traffic to the minikube cluster:

```bash
# First, port forward to minikube
make port-forward
```

Then, we can test them via `curl`:

```bash
# Health endpoint - confirms service is responding
$ curl -f http://localhost:8080/health
{"status":"healthy","model_loaded":true,"version":"1.0.0"}

# Readiness endpoint - confirms model is loaded and ready
$ curl -f http://localhost:8080/ready
{"status":"ready"}

# Model info endpoint - validates model metadata
$ curl http://localhost:8080/model-info
{"model_type":"RandomForestClassifier","n_features":20,"n_classes":2,"classes":[0,1]}
```

Functional Testing:
```bash
# Single prediction test
$ curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0]]
  }'

{"predictions":[0],"probabilities":[[0.6,0.4]]}

# Batch prediction
$ curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [
      [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
      [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0]
    ]
  }'

{"predictions":[0,0],"probabilities":[[0.6,0.4],[0.6,0.4]]}
```

Load Testing:
```bash
# Concurrent request testing
$ for i in {1..10}; do
  curl -s -X POST http://localhost:8080/predict \
    -H "Content-Type: application/json" \
    -d '{"features": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0]]}' &
done
$ wait
[2] 51061
[3] 51062
[4] 51063
[5] 51064
[6] 51065
[7] 51066
[8] 51067
[9] 51068
[10] 51069
[11] 51070
{"predictions":[0],"probabilities":[[0.6,0.4]]}[4]    done       curl -s -X POST http://localhost:8080/predict -H  -d
{"predictions":[0],"probabilities":[[0.6,0.4]]}[7]    done       curl -s -X POST http://localhost:8080/predict -H  -d
{"predictions":[0],"probabilities":[[0.6,0.4]]}[6]    done       curl -s -X POST http://localhost:8080/predict -H  -d
{"predictions":[0],"probabilities":[[0.6,0.4]]}[5]    done       curl -s -X POST http://localhost:8080/predict -H  -d
{"predictions":[0],"probabilities":[[0.6,0.4]]}[2]    done       curl -s -X POST http://localhost:8080/predict -H  -d
{"predictions":[0],"probabilities":[[0.6,0.4]]}[3]    done       curl -s -X POST http://localhost:8080/predict -H  -d
{"predictions":[0],"probabilities":[[0.6,0.4]]}[10]  - done       curl -s -X POST http://localhost:8080/predict -H  -d
{"predictions":[0],"probabilities":[[0.6,0.4]]}[8]    done       curl -s -X POST http://localhost:8080/predict -H  -d
{"predictions":[0],"probabilities":[[0.6,0.4]]}[9]  - done       curl -s -X POST http://localhost:8080/predict -H  -d
{"predictions":[0],"probabilities":[[0.6,0.4]]}[11]  + done       curl -s -X POST http://localhost:8080/predict -H  -d
```

Note that all of the predictions are the same since we're just using a
dummy model with scikit-learn.

#### Volcano-Specific Validation:

We can then view the volcano-specific kubernetes objects:

```bash
# Verify Volcano Queues
$ kubectl get queue -o wide
NAME       PARENT
default    root
ml-queue   root
root

# Verify Volcano job status
$ kubectl get vcjob ml-model-job -n ml-models
NAME           STATUS    MINAVAILABLE   RUNNINGS   AGE
ml-model-job   Running   2              2          10m

# Check gang scheduling worked correctly
# Should show all pods started simultaneously
$ kubectl get pods -l app=ml-model -n ml-models -o wide
NAME                       READY   STATUS    RESTARTS   AGE     IP            NODE       NOMINATED NODE   READINESS GATES
ml-model-job-ml-server-0   1/1     Running   0          9m24s   10.244.0.10   minikube   <none>           <none>
ml-model-job-ml-server-1   1/1     Running   0          9m24s   10.244.0.11   minikube   <none>           <none>
```

We should see that two pods were started at the same time (gang scheduling).

## Customization

### Using Your Own Model

1. Replace the sample model creation in `app.py`
2. Update the feature count and model loading logic
3. Rebuild the Docker image
4. Update resource requirements if needed

### Scaling

To adjust the number of replicas:

1. Edit `k8s/ml-model-deployment.yaml`
1. Change `replicas` (2) to your desired number
1. Also update `minAvailable` in the Volcano job spec

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

---

This example uses a simple scikit-learn model for demonstration.
In production, replace with your actual ML model and adjust resource
requirements accordingly.
For all of the relevant code, please visit my
[GitHub repo](https://github.com/afoley587/coding-challenges-2025/tree/main/machine-learning/volcano)!
