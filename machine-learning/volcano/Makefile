# Make all targets PHONY
MAKEFLAGS += --always-make

IMAGE_NAME := ml-model
IMAGE_TAG := latest
NAMESPACE := ml-models
VOLCANO_VERSION := v1.12.0
HELM_RELEASE_NAME := volcano-system

# Default target
.DEFAULT_GOAL := help

help:  ## Show this help.
	@egrep -h '\s##\s' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m  %-30s\033[0m %s\n", $$1, $$2}'

build: ## Build the Docker image
	@echo "Building Docker image..."
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) ./python/
	@echo "Image built successfully: $(IMAGE_NAME):$(IMAGE_TAG)"

test-local: build ## Test the application locally
	@echo "Running local test..."
	docker run --rm -p 8000:8000 --name ml-model-test $(IMAGE_NAME):$(IMAGE_TAG) &
	sleep 10
	curl -f http://localhost:8000/health || (docker stop ml-model-test && exit 1)
	curl -f http://localhost:8000/ready || (docker stop ml-model-test && exit 1)
	docker stop ml-model-test
	@echo "Local test completed successfully"

minikube-start: ## Start minikube with appropriate resources
	@echo "Starting minikube..."
	minikube start --cpus=4 --memory=4096 --disk-size=20g
	minikube addons enable ingress
	@echo "Minikube started successfully"

minikube-build: ## Build image in minikube Docker environment
	@echo "Building image in minikube..."
	eval $$(minikube docker-env) && docker build -t $(IMAGE_NAME):$(IMAGE_TAG) ./python/
	@echo "Image built in minikube environment"

minikube-stop: ## Stop minikube
	minikube stop

install-volcano: ## Install Volcano scheduler using Helm
	@echo "Installing Volcano scheduler..."
	helm repo add volcano-sh https://volcano-sh.github.io/helm-charts
	helm repo update
	kubectl create namespace volcano-system --dry-run=client -o yaml | kubectl apply -f -
	helm upgrade --install $(HELM_RELEASE_NAME) volcano-sh/volcano \
		--namespace volcano-system \
		--version $(VOLCANO_VERSION) \
		--set basic.image.tag=$(VOLCANO_VERSION) \
		--wait --timeout=300s
	@echo "Waiting for Volcano to be ready..."
	kubectl wait --for=condition=ready pod -l app=volcano-scheduler --namespace volcano-system --timeout=300s
	@echo "Volcano installed successfully"

verify-volcano: ## Verify Volcano installation
	@echo "Verifying Volcano installation..."
	kubectl get pods -n volcano-system
	kubectl get crd | grep volcano
	@echo "Volcano verification completed"

uninstall-volcano: ## Uninstall Volcano scheduler
	@echo "Uninstalling Volcano..."
	helm uninstall $(HELM_RELEASE_NAME) --namespace volcano-system
	kubectl delete namespace volcano-system
	@echo "Volcano uninstalled"

create-namespace: ## Create application namespace
	@echo "Creating namespace..."
	kubectl create namespace $(NAMESPACE) --dry-run=client -o yaml | kubectl apply -f -

deploy-queues: create-namespace ## Deploy Volcano queues
	@echo "Deploying Volcano queues..."
	kubectl apply -f k8s/volcano-queue.yaml

deploy-app: create-namespace deploy-queues ## Deploy the ML model application
	@echo "Deploying ML model application..."
	kubectl apply -f k8s/ml-model-deployment.yaml
	@echo "Application deployed successfully"

test-deployment: ## Test the deployed application
	@echo "Testing deployment..."
	kubectl get pods -n $(NAMESPACE)
	kubectl get services -n $(NAMESPACE)
	@echo "Port-forwarding to test the service..."
	kubectl port-forward service/ml-model-service 8080:80 -n $(NAMESPACE) &
	sleep 5
	curl -f http://localhost:8080/health || (pkill -f "kubectl port-forward" && exit 1)
	curl -f http://localhost:8080/ready || (pkill -f "kubectl port-forward" && exit 1)
	@echo "Testing prediction endpoint..."
	curl -X POST http://localhost:8080/predict \
		-H "Content-Type: application/json" \
		-d '{"features": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0]]}' || (pkill -f "kubectl port-forward" && exit 1)
	pkill -f "kubectl port-forward"
	@echo "Deployment test completed successfully"

load-test: ## Run a simple load test
	@echo "Running load test..."
	kubectl port-forward service/ml-model-service 8080:80 -n $(NAMESPACE) &
	sleep 5
	for i in $$(seq 1 10); do \
		curl -s -X POST http://localhost:8080/predict \
			-H "Content-Type: application/json" \
			-d '{"features": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0]]}' & \
	done
	wait
	pkill -f "kubectl port-forward"
	@echo "Load test completed"

logs: ## View application logs
	kubectl logs -l app=ml-model -n $(NAMESPACE) --tail=50 -f

describe-job: ## Describe Volcano job
	kubectl describe vcjob ml-model-job -n $(NAMESPACE)

get-events: ## Get cluster events
	kubectl get events -n $(NAMESPACE) --sort-by=.metadata.creationTimestamp

clean-app: ## Remove the application
	@echo "Cleaning up application..."
	kubectl delete -f k8s/ml-model-deployment.yaml --ignore-not-found=true
	kubectl delete -f k8s/volcano-queue.yaml --ignore-not-found=true
	@echo "Application cleaned up"

clean-namespace: ## Delete the namespace
	kubectl delete namespace $(NAMESPACE) --ignore-not-found=true

clean-all: clean-app clean-namespace ## Clean up everything
	@echo "Cleanup completed"

minikube-deploy: minikube-start install-volcano minikube-build deploy-app test-deployment ## Complete minikube deployment
	@echo "Complete minikube deployment finished successfully"
	@echo "You can access the service at: http://localhost:8080 (after port-forwarding)"
	@echo "Run 'make logs' to view application logs"
	@echo "Run 'make clean-all' to clean up"

production-deploy: build install-volcano deploy-app test-deployment ## Complete production deployment
	@echo "Production deployment completed successfully"

shell: ## Open a shell in a running pod
	kubectl exec -it $$(kubectl get pods -l app=ml-model -n $(NAMESPACE) -o jsonpath='{.items[0].metadata.name}') -n $(NAMESPACE) -- /bin/bash

port-forward: ## Port forward to the service
	@echo "Port forwarding to http://localhost:8080"
	kubectl port-forward service/ml-model-service 8080:80 -n $(NAMESPACE)

get-status: ## Get deployment status
	@echo "=== Namespace ==="
	kubectl get namespace $(NAMESPACE) 2>/dev/null || echo "Namespace not found"
	@echo "\n=== Volcano Jobs ==="
	kubectl get vcjob -n $(NAMESPACE) 2>/dev/null || echo "No Volcano jobs found"
	@echo "\n=== Pods ==="
	kubectl get pods -l app=ml-model -n $(NAMESPACE) 2>/dev/null || echo "No pods found"
	@echo "\n=== Services ==="
	kubectl get services -n $(NAMESPACE) 2>/dev/null || echo "No services found"
	@echo "\n=== Ingress ==="
	kubectl get ingress -n $(NAMESPACE) 2>/dev/null || echo "No ingress found"
