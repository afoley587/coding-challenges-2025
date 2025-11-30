package cmd

import (
	"fmt"
	"github.com/spf13/cobra"
	"os"
)

// rootCmd is the base command for the CLI.  It delegates to
// subcommands defined in client.go and server.go.  See init
// functions in those files for flag definitions.
var rootCmd = &cobra.Command{
	Use:   "grpc-golang",
	Short: "CLI for the gRPC Golang API example",
	Long:  "Command line interface to run the gRPC server and interact with it as a client.",
}

// Execute runs the root command.  It should be invoked from main.
func Execute() {
	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}
