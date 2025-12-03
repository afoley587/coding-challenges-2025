package client

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"io"
	"os"
	"time"

	pb "github.com/afoley587/coding-challenges-2025/grpc-golang-api/proto"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/protobuf/types/known/emptypb"
)

// dial creates a gRPC client connection to the server at addr.
// The returned connection must be closed by
// the caller.
func dial(cfg DialConfig) (*grpc.ClientConn, error) {
	var opts grpc.DialOption

	if cfg.Insecure {
		opts = grpc.WithTransportCredentials(insecure.NewCredentials())
	} else {
		creds, err := loadTLSCredentials(cfg)
		if err != nil {
			return nil, fmt.Errorf("failed to load TLS credentials: %w", err)
		}
		opts = grpc.WithTransportCredentials(creds)
	}

	return grpc.Dial(cfg.Address, opts)
}

func loadTLSCredentials(cfg DialConfig) (credentials.TransportCredentials, error) {
	tlsCfg := &tls.Config{}

	if cfg.RootCA != "" {
		pem, err := os.ReadFile(cfg.RootCA)
		if err != nil {
			return nil, fmt.Errorf("reading root CA: %w", err)
		}

		pool := x509.NewCertPool()
		if !pool.AppendCertsFromPEM(pem) {
			return nil, fmt.Errorf("failed to add root CA to cert pool")
		}
		tlsCfg.RootCAs = pool
	}

	if cfg.ClientCert != "" && cfg.ClientKey != "" {
		clientCert, err := tls.LoadX509KeyPair(cfg.ClientCert, cfg.ClientKey)
		if err != nil {
			return nil, fmt.Errorf("loading client certificate: %w", err)
		}
		tlsCfg.Certificates = []tls.Certificate{clientCert}
	}

	return credentials.NewTLS(tlsCfg), nil
}

// ListUsers connects to the gRPC server at addr and prints all users
// to stdout.  It returns an error if the request fails.
func (c *GRPCClient) ListUsers() error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	stream, err := c.Client().ListUsers(ctx, &emptypb.Empty{})
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
func (c *GRPCClient) CreateUser(name, email string) (*pb.User, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	req := &pb.CreateUserRequest{
		Name:  &name,
		Email: &email,
	}

	return c.Client().CreateUser(ctx, req)
}

// GetUser connects to the gRPC server at addr and retrieves a user by id.
// It returns the user or nil if not found.  Any RPC error is
// returned to the caller.
func (c *GRPCClient) GetUser(id int32) (*pb.User, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	req := &pb.GetUserRequest{Id: &id}

	return c.Client().GetUser(ctx, req)
}
