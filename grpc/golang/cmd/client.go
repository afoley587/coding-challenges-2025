package cmd

import (
	"errors"
	"fmt"
	"github.com/spf13/cobra"

	"github.com/afoley587/coding-challenges-2025/grpc-golang-api/internal/client"
)

var (
	clientServerAddr string
	newName          string
	newEmail         string
	userId           int32
)

// clientCmd groups client subcommands.
var clientCmd = &cobra.Command{
	Use:   "client",
	Short: "Interact with the gRPC server",
	Long:  "Commands for listing, creating and retrieving users via the gRPC client.",
}

// listCmd lists all users in the server.
var listCmd = &cobra.Command{
	Use:   "list",
	Short: "List all users",
	RunE: func(cmd *cobra.Command, args []string) error {
		return client.ListUsers(clientServerAddr)
	},
}

// createCmd creates a new user on the server.
var createCmd = &cobra.Command{
	Use:   "create",
	Short: "Create a new user",
	RunE: func(cmd *cobra.Command, args []string) error {
		if newName == "" || newEmail == "" {
			return errors.New("both --name and --email must be specified")
		}
		user, err := client.CreateUser(clientServerAddr, newName, newEmail)
		if err != nil {
			return err
		}
		fmt.Printf("Created user: %v\n", user)
		return nil
	},
}

// getCmd retrieves a user by id.
var getCmd = &cobra.Command{
	Use:   "get",
	Short: "Get a user by ID",
	RunE: func(cmd *cobra.Command, args []string) error {
		user, err := client.GetUser(clientServerAddr, userId)
		if err != nil {
			return err
		}
		if user == nil {
			fmt.Println("User not found")
			return nil
		}
		fmt.Printf("%v\n", user)
		return nil
	},
}

func init() {
	// persistent flags apply to all subcommands
	clientCmd.PersistentFlags().StringVarP(&clientServerAddr, "addr", "a", "127.0.0.1:9090", "Server address")
	// flags specific to subcommands
	createCmd.Flags().StringVarP(&newName, "name", "n", "", "Name of the user")
	createCmd.Flags().StringVarP(&newEmail, "email", "e", "", "Email of the user")
	getCmd.Flags().Int32VarP(&userId, "id", "i", 0, "ID of the user to retrieve")

	clientCmd.AddCommand(listCmd)
	clientCmd.AddCommand(createCmd)
	clientCmd.AddCommand(getCmd)
	rootCmd.AddCommand(clientCmd)
}
