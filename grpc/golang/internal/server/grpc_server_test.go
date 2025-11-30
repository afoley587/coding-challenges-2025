package server_test

import (
    "context"
    "io"
    "net"
    "testing"
    "time"

    pb "github.com/afoley587/coding-challenges-2025/grpc-golang-api/proto"
    srv "github.com/afoley587/coding-challenges-2025/grpc-golang-api/internal/server"
    "github.com/afoley587/coding-challenges-2025/grpc-golang-api/internal/store"
    "google.golang.org/grpc"
    "google.golang.org/grpc/credentials/insecure"
    "google.golang.org/protobuf/types/known/emptypb"
)

// startTestServer spins up a gRPC server on a random local port backed by
// the provided store.  It returns the server address and a shutdown
// function to be deferred by the caller.
func startTestServer(t *testing.T, us store.UserStore) (addr string, shutdown func()) {
    t.Helper()
    lis, err := net.Listen("tcp", "127.0.0.1:0")
    if err != nil {
        t.Fatalf("failed to listen: %v", err)
    }
    grpcServer := grpc.NewServer()
    pb.RegisterGrpcGolangAPIServer(grpcServer, srv.NewGRPCServer(us))
    go func() {
        _ = grpcServer.Serve(lis)
    }()
    return lis.Addr().String(), func() {
        grpcServer.Stop()
        _ = lis.Close()
    }
}

// stringPtr returns a pointer to the provided string.  It is used to
// construct request messages with non‑nil pointer fields.
func stringPtr(s string) *string { return &s }

// int32Ptr returns a pointer to the provided int32.
func int32Ptr(i int32) *int32 { return &i }

// TestGrpcService exercises the CreateUser, GetUser and ListUsers RPCs
// against an in‑memory backed server.  It verifies the end‑to‑end
// behavior through gRPC rather than invoking the store directly.
func TestGrpcService(t *testing.T) {
    s := store.NewInMemoryStore()
    addr, shutdown := startTestServer(t, s)
    defer shutdown()

    conn, err := grpc.Dial(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
    if err != nil {
        t.Fatalf("failed to dial server: %v", err)
    }
    defer conn.Close()
    client := pb.NewGrpcGolangAPIClient(conn)

    ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
    defer cancel()

    // Create a new user.
    cu, err := client.CreateUser(ctx, &pb.CreateUserRequest{Name: stringPtr("Bob"), Email: stringPtr("bob@example.com")})
    if err != nil {
        t.Fatalf("CreateUser RPC failed: %v", err)
    }
    if cu.GetId() != 0 {
        t.Fatalf("expected created user id 0, got %d", cu.GetId())
    }

    // Retrieve the user by ID.
    gu, err := client.GetUser(ctx, &pb.GetUserRequest{Id: int32Ptr(0)})
    if err != nil {
        t.Fatalf("GetUser RPC failed: %v", err)
    }
    if gu.GetName() != "Bob" || gu.GetEmail() != "bob@example.com" {
        t.Fatalf("GetUser returned unexpected user: %+v", gu)
    }

    // List users and ensure exactly one user is streamed.
    stream, err := client.ListUsers(ctx, &emptypb.Empty{})
    if err != nil {
        t.Fatalf("ListUsers RPC failed: %v", err)
    }
    count := 0
    for {
        user, err := stream.Recv()
        if err == io.EOF {
            break
        }
        if err != nil {
            t.Fatalf("stream Recv error: %v", err)
        }
        count++
        if user.GetName() != "Bob" {
            t.Fatalf("expected streamed user name Bob, got %s", user.GetName())
        }
    }
    if count != 1 {
        t.Fatalf("expected 1 user from ListUsers, got %d", count)
    }
}