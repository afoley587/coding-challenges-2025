# Make all targets PHONY
MAKEFLAGS += --always-make

help:  ## Show this help.
	@egrep -h '\s##\s' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m  %-30s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help

build: proto  ## Build golang application
	go build

clean:  ## Removes build application binaries
	go clean

docker:  ## Build docker images
	docker build . -t afoley587/coding-challenges/grpc-golang:latest

fmt:  ## Run go fmt to format code
	go fmt

minikube:  ## Start a local minikube cluster
	minikube start

proto:  ## Generate protocol buffer code
	protoc --go_out=. --go_opt=paths=source_relative \
		--go-grpc_out=. --go-grpc_opt=paths=source_relative \
		proto/users.proto

run-server:  ## Runs the server-side application
	go run . server run \
		--addr 0.0.0.0:9090 \
		--redis-address 127.0.0.1:6379

skaffold:  ## Runs the application with skaffold
	skaffold dev

test:  ## Runs all go tests
	go test ./...
