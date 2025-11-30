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

## Generating gRPC code

To regenerate the Go stubs from the `proto/users.proto` definition you
need `protoc` with the Go and gRPC plugins. 
Run the following commands from within the `grpc/golang` directory:

```bash
make proto
```

This will update the generated code under the `proto` package.

## Building

Ensure you have a recent version of Go installed (1.22 or later) and
run:

```bash
make build
```

## Running the server

Start a local Redis instance on the default port, then run:

```bash
go run . server run \
    --addr 0.0.0.0:9090 \
    --redis-address 127.0.0.1:6379
```

This will start the gRPC server listening on port 9090.  You can
override the address or Redis endpoint via flags.

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
