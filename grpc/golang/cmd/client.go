package cmd

import (
	"fmt"

    "github.com/spf13/cobra"
	"github.com/afoley587/coding-challenges-2025/grpc-golang-api/internal/client"
)

var serverAddr string

var clientCmd = &cobra.Command{
    Use:     "client",
    Aliases: []string{},
    Short:   "Client commands.",
    Long:    "Commands related to listing, creating, or deleting users via client.",
}

var listUsersCmd = &cobra.Command{
	Use:   "list-users",
	Short: "list users in the database",
	Run: func(cmd *cobra.Command, args []string) {
		fmt.Println("Creating a new birthday entry...")
		listUsers(serverAddr)
	},
}

var createUserCmd = &cobra.Command{
	Use:   "create-users",
	Short: "create users in the database",
	Run: func(cmd *cobra.Command, args []string) {
		fmt.Println("Creating a new birthday entry...")
		createUser(serverAddr)
	},
}


func init() {
    clientCmd.Flags().StringVarP(&serverAddr, "server-addr", "s", "127.0.0.1:9090", "Server address to connect to")
	clientCmd.AddCommand(listUsersCmd)
	clientCmd.AddCommand(createUserCmd)
    rootCmd.AddCommand(clientCmd)
}

func listUsers(serverAddr string) {
	client.ListUsers(serverAddr)
}

func createUser(serverAddr string) {
	client.CreateUser(serverAddr)
}