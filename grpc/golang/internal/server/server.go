package server

import (
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"net"
	"os"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"

	"github.com/afoley587/coding-challenges-2025/grpc-golang-api/internal/store"
	pb "github.com/afoley587/coding-challenges-2025/grpc-golang-api/proto"
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

// RunTLS starts a gRPC server with mTLS enabled.
// certFile and keyFile are the server's certificate and private key.
// caFile contains one or more CA certificates used to verify client certs.
func RunTLS(addr string, certFile string, keyFile string, caFile string, store store.UserStore) error {
	serverCert, err := tls.LoadX509KeyPair(certFile, keyFile)
	if err != nil {
		return fmt.Errorf("failed loading server key pair: %w", err)
	}

	caPem, err := os.ReadFile(caFile)
	if err != nil {
		return fmt.Errorf("failed reading CA file: %w", err)
	}
	clientCAs := x509.NewCertPool()
	if ok := clientCAs.AppendCertsFromPEM(caPem); !ok {
		return fmt.Errorf("failed parsing CA file")
	}

	tlsConf := &tls.Config{
		Certificates: []tls.Certificate{serverCert},
		ClientCAs:    clientCAs,
		ClientAuth:   tls.RequireAndVerifyClientCert,
	}
	creds := credentials.NewTLS(tlsConf)

	lis, err := net.Listen("tcp", addr)
	if err != nil {
		return fmt.Errorf("failed to listen: %w", err)
	}
	grpcServer := grpc.NewServer(grpc.Creds(creds))
	pb.RegisterGrpcGolangAPIServer(grpcServer, NewGRPCServer(store))
	return grpcServer.Serve(lis)
}
