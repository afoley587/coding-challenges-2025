package server

import (
	"fmt"
	"net"

	"github.com/afoley587/coding-challenges-2025/grpc-golang-api/internal/store"
	pb "github.com/afoley587/coding-challenges-2025/grpc-golang-api/proto"
	"google.golang.org/grpc"
)

// Run starts a gRPC server listening on addr using the provided
// UserStore.  It registers the generated service implementation and
// blocks until the server stops serving.  Any error encountered while
// starting or serving will be returned.
func Run(addr string, s store.UserStore) error {
	lis, err := net.Listen("tcp", addr)
	if err != nil {
		return fmt.Errorf("failed to listen on %s: %w", addr, err)
	}
	grpcServer := grpc.NewServer()
	pb.RegisterGrpcGolangAPIServer(grpcServer, NewGRPCServer(s))
	return grpcServer.Serve(lis)
}
