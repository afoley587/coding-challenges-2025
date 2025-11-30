package client

import (
	"context"
	"fmt"
	"io"
	"time"

	pb "github.com/afoley587/coding-challenges-2025/grpc-golang-api/proto"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/protobuf/types/known/emptypb"
)

// dial creates a gRPC client connection to the server at addr using
// insecure credentials.  The returned connection must be closed by
// the caller.
func dial(addr string) (*grpc.ClientConn, error) {
	return grpc.Dial(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
}

// ListUsers connects to the gRPC server at addr and prints all users
// to stdout.  It returns an error if the request fails.
func ListUsers(addr string) error {
	conn, err := dial(addr)
	if err != nil {
		return fmt.Errorf("failed to dial server: %w", err)
	}
	defer conn.Close()
	c := pb.NewGrpcGolangAPIClient(conn)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	stream, err := c.ListUsers(ctx, &emptypb.Empty{})
	if err != nil {
		return fmt.Errorf("ListUsers RPC failed: %w", err)
	}
	for {
		user, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("error receiving user: %w", err)
		}
		fmt.Printf("%v\n", user)
	}
	return nil
}

// CreateUser connects to the gRPC server at addr and creates a new
// user with the provided name and email.  It returns the created
// user or an error.
func CreateUser(addr, name, email string) (*pb.User, error) {
	conn, err := dial(addr)
	if err != nil {
		return nil, fmt.Errorf("failed to dial server: %w", err)
	}
	defer conn.Close()
	c := pb.NewGrpcGolangAPIClient(conn)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	req := &pb.CreateUserRequest{
		Name:  &name,
		Email: &email,
	}
	user, err := c.CreateUser(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("CreateUser RPC failed: %w", err)
	}
	return user, nil
}

// GetUser connects to the gRPC server at addr and retrieves a user by id.
// It returns the user or nil if not found.  Any RPC error is
// returned to the caller.
func GetUser(addr string, id int32) (*pb.User, error) {
	conn, err := dial(addr)
	if err != nil {
		return nil, fmt.Errorf("failed to dial server: %w", err)
	}
	defer conn.Close()
	c := pb.NewGrpcGolangAPIClient(conn)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	req := &pb.GetUserRequest{Id: &id}
	user, err := c.GetUser(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("GetUser RPC failed: %w", err)
	}
	return user, nil
}
