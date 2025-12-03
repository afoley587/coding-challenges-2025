# Make all targets PHONY
MAKEFLAGS += --always-make

help:  ## Show this help.
	@egrep -h '\s##\s' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m  %-30s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help

build:  ## Build golang application
	@go build

clean:  ## Removes build application binaries
	@go clean

certs:  ## Generates the mTLS certs
	@echo "Generating TLS and mTLS certificates..."
	@bash scripts/generate-certs.sh
	@echo "Done."

docker:  ## Build docker images
	@docker build . -t afoley587/coding-challenges/grpc-golang:latest

fmt:  ## Run go fmt to format code
	@go fmt

minikube:  ## Start a local minikube cluster
	@minikube start

proto:  ## Generate protocol buffer code
	@protoc --go_out=. --go_opt=paths=source_relative \
		--go-grpc_out=. --go-grpc_opt=paths=source_relative \
		proto/users.proto

redis:  ## Start a redis instance
	@if ! docker ps --filter=name=redis | grep redis; then \
		echo "Starting redis"; \
		docker run -t -d -p6379:6379 --name redis redis; \
	fi;

run-client-mtls:  ## Runs a few test client commands with mTLS enabled
	@echo "Listing Users"

	@go run . client list \
		--addr localhost:9090 \
		--tls-cert certs/client/client.crt \
		--tls-key certs/client/client.key \
		--tls-ca certs/ca/ca.crt

	@echo "Creating A Test User"

	@go run . client create \
		--addr localhost:9090 \
		--name Alice \
		--email alice@example.com \
		--tls-cert certs/client/client.crt \
		--tls-key certs/client/client.key \
		--tls-ca certs/ca/ca.crt

	@echo "Getting User with ID 0"

	@go run . client get \
		--addr localhost:9090 \
		--id 0 \
		--tls-cert certs/client/client.crt \
		--tls-key certs/client/client.key \
		--tls-ca certs/ca/ca.crt

run-server-mtls:  ## Runs the server-side application with mTLS enabled
	@go run . server run \
		--addr 0.0.0.0:9090 \
		--redis-address 127.0.0.1:6379 \
		--cert certs/server/server.crt \
		--key certs/server/server.key \
		--ca certs/ca/ca.crt \
		--mtls

run-server-insecure:  ## Runs the server-side application with mTLS disabled
	@go run . server run \
		--addr 0.0.0.0:9090 \
		--redis-address 127.0.0.1:6379


skaffold:  ## Runs the application with skaffold
	@skaffold dev

test:  ## Runs all go tests
	@go test ./...
