package cmd

import (
	"errors"
	"fmt"
	"log"

	"github.com/afoley587/coding-challenges-2025/grpc-golang-api/internal/client"
	"github.com/spf13/cobra"
)

var (
	// server address
	clientServerAddr string

	// create/get fields
	newName  string
	newEmail string
	userId   int32

	// TLS flags
	insecure      bool
	tlsCA         string
	tlsClientCert string
	tlsClientKey  string
)

// Build DialConfig from CLI flags
func getDialConfig() client.DialConfig {
	return client.DialConfig{
		Address:    clientServerAddr,
		Insecure:   insecure,
		RootCA:     tlsCA,
		ClientCert: tlsClientCert,
		ClientKey:  tlsClientKey,
	}
}

// Wrapper to build a high-level client
func getClient() (*client.GRPCClient, error) {
	cfg := getDialConfig()
	return client.NewClient(cfg)
}

// Root client command
var clientCmd = &cobra.Command{
	Use:   "client",
	Short: "Interact with the gRPC server",
	Long:  "Commands for listing, creating, and retrieving users via the gRPC client.",
}

var listCmd = &cobra.Command{
	Use:   "list",
	Short: "List all users",
	RunE: func(cmd *cobra.Command, args []string) error {
		c, err := getClient()
		if err != nil {
			return err
		}
		defer c.Close()
		log.Printf("Listing users from %s", clientServerAddr)
		return c.ListUsers()
	},
}

var createCmd = &cobra.Command{
	Use:   "create",
	Short: "Create a new user",
	RunE: func(cmd *cobra.Command, args []string) error {
		if newName == "" || newEmail == "" {
			return errors.New("both --name and --email must be specified")
		}

		c, err := getClient()
		if err != nil {
			return err
		}
		defer c.Close()
		user, err := c.CreateUser(newName, newEmail)
		if err != nil {
			return err
		}
		fmt.Printf("Created user: %v\n", user)
		return nil
	},
}

var getCmd = &cobra.Command{
	Use:   "get",
	Short: "Get a user by ID",
	RunE: func(cmd *cobra.Command, args []string) error {
		c, err := getClient()
		if err != nil {
			return err
		}
		defer c.Close()

		user, err := c.GetUser(userId)
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

	clientCmd.PersistentFlags().StringVarP(&clientServerAddr,
		"addr", "a", "127.0.0.1:9090", "Server address")

	clientCmd.PersistentFlags().BoolVar(
		&insecure, "insecure", false, "Use insecure gRPC (no TLS)")

	clientCmd.PersistentFlags().StringVar(
		&tlsCA, "tls-ca", "", "Path to root CA certificate")

	clientCmd.PersistentFlags().StringVar(
		&tlsClientCert, "tls-cert", "", "Path to client certificate for mTLS")

	clientCmd.PersistentFlags().StringVar(
		&tlsClientKey, "tls-key", "", "Path to client private key for mTLS")

	createCmd.Flags().StringVarP(&newName, "name", "n", "", "Name of the user")

	createCmd.Flags().StringVarP(&newEmail, "email", "e", "", "Email of the user")

	getCmd.Flags().Int32VarP(&userId, "id", "i", 0, "ID of the user to retrieve")

	clientCmd.AddCommand(listCmd, createCmd, getCmd)
	rootCmd.AddCommand(clientCmd)
}
