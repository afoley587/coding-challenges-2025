package cmd

import (
	"log"
	"net"

    "github.com/spf13/cobra"

	pb "github.com/afoley587/coding-challenges-2025/grpc-golang-api/proto"

	"github.com/afoley587/coding-challenges-2025/grpc-golang-api/internal/server"
	"google.golang.org/grpc"
)

var listenAddr string
var redisAddr string

var serverCmd = &cobra.Command{
    Use:     "server",
    Aliases: []string{},
    Short:   "Server commands.",
    Long:    "Commands related to running the server.",
}

var runServerCmd = &cobra.Command{
	Use:   "run-server",
	Short: "Run the server",
	Run: func(cmd *cobra.Command, args []string) {
		runServer(listenAddr, redisAddr)
	},
}

func init() {
    serverCmd.Flags().StringVarP(&listenAddr, "server-addr", "s", "0.0.0.0:9090", "Server address to listen on")
    serverCmd.Flags().StringVarP(&redisAddr, "redis-addr", "r", "127.0.0.1:6379", "Redis address to connect to")
	serverCmd.AddCommand(runServerCmd)
    rootCmd.AddCommand(serverCmd)
}


func runServer(listenAddr string, redisAddr string) {
	lis, err := net.Listen("tcp", listenAddr)

	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}
	grpcServer := grpc.NewServer()
	pb.RegisterGrpcGolangAPIServer(grpcServer, server.NewServer(redisAddr))
	log.Println("Registered")
	grpcServer.Serve(lis)
}
