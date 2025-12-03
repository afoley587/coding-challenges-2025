# Coding Challenges: Production-Ready gRPC API
## Building a gRPC API in Go with mTLS

Coding challenges are important to undertake.
They help you expand or deepen your knowledge of a language, protocol,
or technology.
This project is a complete example of a **gRPC service written in Go**,
backed by **Redis**, secured with **mTLS**,
and wrapped in a clean **Cobra-based CLI**.
It is designed as both a learning tool and a realistic
foundation for production-style gRPC services.
My goal for this project was to:

1. Deepen my `golang` knowledge

1. Deepen my knowledge of structuring production projects

1. Deepen my `mTLS` knowledge and know-how

The service exposes a simple **User API** with three RPCs:

* **CreateUser** – Create a new user with name and email
* **GetUser** – Retrieve a user by ID
* **ListUsers** – Stream all users from the store

Although the data model is small, the project demonstrates
production-grade patterns, including:

* gRPC server and client structure
* Redis as a pluggable persistence backend
* Mutual TLS (mTLS)
* Graceful CLI tooling with Cobra
* Skaffold workflows for local Kubernetes development

As a disclaimer, I won't be going into depth about the code
in this project.
If you want to go into depth and inspect the code, please view it on
[GitHub](https://github.com/afoley587/coding-challenges-2025/tree/main/grpc/golang).

## Table Of Contents

<!-- toc -->

- [Why gRPC?](#why-grpc)
- [What is mTLS?](#what-is-mtls)
- [Project Structure](#project-structure)
- [Features](#features)
- [Pre-Requisites](#pre-requisites)
- [Generating gRPC Code](#generating-grpc-code)
- [Building the Application](#building-the-application)
- [Running the Service with mTLS](#running-the-service-with-mtls)
- [Running With Skaffold](#running-with-skaffold)
- [Running the Client (Local, Without mTLS)](#running-the-client-local-without-mtls)
- [Testing](#testing)
- [Conclusion](#conclusion)

<!-- tocstop -->

---

## Why gRPC?

gRPC is a high-performance RPC framework built on HTTP/2.
It provides:

* Strongly typed client/server contracts using Protocol Buffers
* Bi-directional streaming
* Automatic generation of client libraries
* Lower latency than REST
* Built-in support for mTLS

This makes it an ideal foundation for microservices,
internal APIs, and high-performance workloads.

---

## What is mTLS?

Traditional TLS provides **server authentication**:
the client verifies the server’s certificate.

**Mutual TLS (mTLS)** goes further:
*both the server and client authenticate each other* using certificates.

This is often used in:

* Zero-trust architectures
* Service-to-service communication
* Highly sensitive internal systems

In this project:

* mTLS mode additionally requires the server to validate client certificates

The CLI and server both support choosing mTLS or plaintext.

---

## Project Structure

```
.
├── cmd/               # Cobra CLI commands (client + server)
├── internal/
│   ├── server/        # gRPC server setup and mTLS configuration
│   ├── store/         # Redis and in-memory user stores
│   └── client/        # gRPC client implementation
├── proto/             # Protobuf definitions and generated code
├── scripts/           # Certificate generation utilities
└── Makefile
```

---

## Features

* gRPC server with user management RPCs
* Pluggable datastore: Redis or in-memory
* mTLS support (server and client)
* Redis password support
* Clean, extensible CLI
* Skaffold workflow for Kubernetes development
* Automated certificate generation via Makefile

The `Makefile` contains multiple commands to help you get up and running:

```bash
$ make help
  help                           Show this help.
  build                          Build golang application
  clean                          Removes build application binaries
  certs                          Generates the mTLS certs
  docker                         Build docker images
  fmt                            Run go fmt to format code
  minikube                       Start a local minikube cluster
  proto                          Generate protocol buffer code
  redis                          Start a redis instance
  run-client-mtls                Runs a few test client commands with mTLS enabled
  run-server-mtls                Runs the server-side application with mTLS enabled
  run-server-insecure            Runs the server-side application with mTLS disabled
  skaffold                       Runs the application with skaffold
  test                           Runs all go tests
```

---

## Pre-Requisites

You should have the following installed:

1. [Docker](https://docs.docker.com/engine/install/)
1. [Go](https://go.dev/doc/install) (1.24+) *
1. [Protoc](https://grpc.io/docs/protoc-installation/)
1. [Protoc Golang Plugin](https://grpc.io/docs/languages/go/quickstart/)
1. [Minikube](https://minikube.sigs.k8s.io/docs/start/) *
1. [Kubectl](https://kubernetes.io/docs/tasks/tools/) *
1. [Skaffold](https://skaffold.dev/docs/install/) *

Tools marked with an asterisk can be installed using `asdf`.

To install via
[`asdf`](https://asdf-vm.com/guide/getting-started.html), clone the
GitHub repository and naviagate to the `grpc/golang directory`.
Then run:

```bash
for plugin in $(awk '{ print $1 }' ../../.tool-versions); do asdf plugin add "$plugin"; done
asdf install
asdf reshim
```

---

## Generating gRPC Code

Once `protoc` is installed, regenerate stubs with:

```bash
make proto
```

This updates the generated Go code in the `proto` directory.

---

## Building the Application

The CLI binary includes both **client** and **server** subcommands.

Build locally:

```bash
make build
```

Or build a Docker image:

```bash
make docker
```

---

## Running the Service with mTLS

We can start a local Redis instance listening on port 6379 with:

```bash
make redis
```

We can generate CA, server, and client certificates with the below.
Both server and client use the certificates under the certs/ directory.

```bash
make certs
```

Then we can run the server using the certificates generated above.
This will start the server with mTLS enabled:
```bash
make run-server-mtls
```

Use the client with mTLS:

```bash
make run-client-mtls
```

Both server and client use the certificates under the `certs/` directory.

---

## Running With Skaffold

For local Kubernetes development, you might want to use `skaffold`.
Using something like `skaffold` allows us to test changes, locally, in real-time.
It standardizes build processes for local development and
enables functions like file-watching and hot-reloading.

```bash
# Start minikube cluster
make minikube
# Start dev stack
make skaffold
```

This runs:

* Redis
* The gRPC server (without mTLS, for dev convenience)

---

## Running the Client (Local, Without mTLS)

List users:

```bash
go run . client list \
    --addr 127.0.0.1:9090
```

Create a user:

```bash
go run . client create \
    --addr 127.0.0.1:9090 \
    --name Alice \
    --email alice@example.com
```

Fetch a user:

```bash
go run . client get \
    --addr 127.0.0.1:9090 \
    --id 0
```

---

## Testing

Unit tests use the in-memory store (no Redis required):

```bash
make test
```

Tests cover:

* Store operations
* gRPC server behavior
* Error handling and validation

---

## Conclusion

This project aims to demonstrate how to build a
well-structured gRPC service in Go with real-world concerns:

* Proper service architecture
* Redis integration
* mTLS
* Clean CLI tooling
* Kubernetes-friendly configuration

It can serve as:

* A learning project
* A template for production microservices
* A reference implementation for secure gRPC communication

Thanks for following along!
Please feel free to copy this code from
[my GitHub repo](https://github.com/afoley587/coding-challenges-2025/tree/main/grpc/golang).
