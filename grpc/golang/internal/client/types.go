package client

import (
	"fmt"
	pb "github.com/afoley587/coding-challenges-2025/grpc-golang-api/proto"
	"google.golang.org/grpc"
)

type DialConfig struct {
	Address    string
	Insecure   bool
	RootCA     string // optional root CA cert
	ClientCert string // optional client cert (mTLS)
	ClientKey  string // optional client key (mTLS)
}

type GRPCClient struct {
	conn *grpc.ClientConn
}

func (c *GRPCClient) Client() pb.GrpcGolangAPIClient {
	return pb.NewGrpcGolangAPIClient(c.conn)
}

func (c *GRPCClient) Close() {
	c.conn.Close()
}

func NewClient(cfg DialConfig) (*GRPCClient, error) {
	conn, err := dial(cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to dial server: %w", err)
	}
	return &GRPCClient{conn: conn}, nil
}
