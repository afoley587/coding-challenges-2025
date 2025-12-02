# gRPC Golang API Example

This directory contains a small example of a gRPC service written in Go.
The service exposes three RPCs for managing users:

* **CreateUser** – create a new user by name and email.
* **GetUser** – retrieve a user by numeric ID.
* **ListUsers** – stream all users to the client.

The server stores users via an abstract `UserStore` interface.
A Redis‑backed store (`internal/store/redis.go`) is used by default, but
an in‑memory implementation (`internal/store/memory.go`) is provided
for tests and development.
The gRPC service itself is implemented in `internal/server/grpc_server.go`
and is registered and run by `internal/server/server.go`.

The command‑line interface is built with
[Cobra](https://github.com/spf13/cobra).
It provides subcommands for running the server and interacting with it
as a client.

## Pre-requisites

It is assumed the following tools are installed:

1. Docker

1. Golang **

1. Minikube **

1. Kubectl **

1. Protoc

1. Skaffold **

> [!TIP]
> ** - This repository's binaries are versioned with
> [`asdf`](https://asdf-vm.com/).
> If using `asdf`, you can automate the installation process with:
>
>     ```bash
>     for plugin in $(awk '{ print $1 }' ../../.tool-versions); do asdf plugin add "$plugin"; done
>     asdf install
>     asdf reshim
>     ```

## Generating gRPC code

To regenerate the Go stubs from the `proto/users.proto` definition you
need `protoc` with the Go and gRPC plugins.
To install `protoc`, please follow the
[installation instructions](https://grpc.io/docs/languages/go/quickstart/)
for your operating system.
Once `protoc` is installed, build the protocol buffers:

```bash
make proto
```

This will update the generated code under the `proto` package.

## Building

Ensure you have a recent version of Go installed (1.24 or later) and
run:

```bash
make build
```

Alternatively, you can build the docker image with

```bash
make docker
```

## Running With Skaffold

The server and redis stack can be run in dev-mode with
[skaffold](https://skaffold.dev/):

```bash
make skaffold
```

This will start the gRPC server listening on port 9090
as well as a redis instance which is not exposed.

## Running the client

List all users:

```bash
go run . client list \
    --addr 127.0.0.1:9090
```

Create a new user:

```bash
go run . client create \
    --addr 127.0.0.1:9090 \
    --name Alice \
    --email alice@example.com
```

Get a user by ID:

```bash
go run . client get \
    --addr 127.0.0.1:9090 \
    --id 0
```

## Testing

Unit tests for the in‑memory store and gRPC service can be run with:

```bash
make test
```

The tests use the in‑memory store to avoid the need
for Redis during testing.
