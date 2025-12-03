package cmd

import (
	"fmt"
	"log"

	"github.com/spf13/cobra"

	"github.com/afoley587/coding-challenges-2025/grpc-golang-api/internal/server"
	"github.com/afoley587/coding-challenges-2025/grpc-golang-api/internal/store"
)

var (
	// server network config
	listenAddr string

	// redis config
	redisAddr     string
	redisPassword string

	// TLS/mTLS flags
	enableMTLS     bool
	serverCertFile string
	serverKeyFile  string
	serverCAFile   string
)

var serverCmd = &cobra.Command{
	Use:   "server",
	Short: "Run the gRPC server",
	Long:  "Commands related to running the gRPC server.",
}

var runServerCmd = &cobra.Command{
	Use:   "run",
	Short: "Start the gRPC server",
	RunE: func(cmd *cobra.Command, args []string) error {

		st, err := store.NewRedisStore(redisAddr, redisPassword, nil)
		if err != nil {
			return fmt.Errorf("redis connection failed: %w", err)
		}

		switch {
		case enableMTLS:
			if serverCertFile == "" || serverKeyFile == "" || serverCAFile == "" {
				return fmt.Errorf("mtls mode requires --cert, --key, and --ca")
			}
			log.Printf("Starting gRPC server with mTLS on %s", listenAddr)
			return server.RunTLS(listenAddr, serverCertFile, serverKeyFile, serverCAFile, st)

		default:
			log.Printf("Starting insecure gRPC server on %s", listenAddr)
			return server.Run(listenAddr, st)
		}
	},
}

func init() {

	runServerCmd.Flags().StringVarP(&listenAddr,
		"addr", "a", "0.0.0.0:9090", "Address to listen on")

	runServerCmd.Flags().StringVarP(&redisAddr,
		"redis-address", "r", "127.0.0.1:6379", "Redis address")

	runServerCmd.Flags().StringVarP(&redisPassword,
		"redis-password", "p", "", "Redis password")

	runServerCmd.Flags().BoolVar(&enableMTLS,
		"mtls", false, "Enable mutual TLS (requires --cert, --key, --ca)")

	runServerCmd.Flags().StringVar(&serverCertFile,
		"cert", "", "Path to server certificate (PEM)")

	runServerCmd.Flags().StringVar(&serverKeyFile,
		"key", "", "Path to server private key (PEM)")

	runServerCmd.Flags().StringVar(&serverCAFile,
		"ca", "", "Path to CA certificate for verifying client certificates (PEM)")

	serverCmd.AddCommand(runServerCmd)
	rootCmd.AddCommand(serverCmd)
}
