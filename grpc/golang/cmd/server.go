package cmd

import (
	"github.com/spf13/cobra"
	"log"

	"github.com/afoley587/coding-challenges-2025/grpc-golang-api/internal/server"
	"github.com/afoley587/coding-challenges-2025/grpc-golang-api/internal/store"
)

var (
	listenAddr string
	redisAddr  string
	redisPassword string
)

// serverCmd is the top‑level command for server operations.
var serverCmd = &cobra.Command{
	Use:   "server",
	Short: "Run the gRPC server",
	Long:  "Commands related to running the gRPC server.",
}

// runServerCmd starts the gRPC server with the specified flags.
var runServerCmd = &cobra.Command{
	Use:   "run",
	Short: "Start the gRPC server",
	RunE: func(cmd *cobra.Command, args []string) error {
		st, err := store.NewRedisStore(redisAddr, redisPassword, nil)
		if err != nil {
			return err
		}
		log.Printf("Starting server on %s (redis: %s)", listenAddr, redisAddr)
		return server.Run(listenAddr, st)
	},
}

func init() {
	runServerCmd.Flags().StringVarP(&listenAddr, "addr", "a", "0.0.0.0:9090", "Address to listen on")
	runServerCmd.Flags().StringVarP(&redisPassword, "redis-password", "p", "", "Redis password")
	runServerCmd.Flags().StringVarP(&redisAddr, "redis-address", "r", "127.0.0.1:6379", "Redis address")
	serverCmd.AddCommand(runServerCmd)
	rootCmd.AddCommand(serverCmd)
}
