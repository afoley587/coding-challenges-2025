# Make all targets PHONY
MAKEFLAGS += --always-make

help:  ## Show this help.
	@egrep -h '\s##\s' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m  %-30s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help

build-arm64:  ## Build golang application (ARM64)
	@GOOS=linux GOARCH=arm64 CGO_ENABLED=0 go build -o bin/custom-bridge-arm64

build-amd64:  ## Build golang application (AMD64)
	@GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go build -o bin/custom-bridge-amd64

build: build-arm64 build-amd64  ## Build golang application

clean:  ## Removes build application binaries
	@go clean

docker:  ## Build docker images
	@docker build . -t afoley587/coding-challenges/custom-cni:latest

fmt:  ## Run go fmt to format code
	@go fmt

minikube-start:  ## Start a local minikube cluster
	@minikube start

skaffold:  ## Runs the application with skaffold
	@skaffold dev

run-test-pod:  ## Runs a test pod and then runs a few commands
	@kubectl run cni-smoke --image=busybox:1.37 --restart=Never --command -- sleep 3600
