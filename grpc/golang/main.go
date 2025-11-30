package main

import "github.com/afoley587/coding-challenges-2025/grpc-golang-api/cmd"

// main is the entry point for the CLI.  It simply delegates to the
// cobra root command defined in the cmd package.
func main() {
	cmd.Execute()
}
