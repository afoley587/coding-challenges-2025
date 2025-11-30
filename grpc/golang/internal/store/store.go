package store

import (
	"context"

	pb "github.com/afoley587/coding-challenges-2025/grpc-golang-api/proto"
)

// UserStore defines an interface for persisting and retrieving users.
//
// Implementations may use different backends (e.g. in‑memory for tests,
// Redis for production).  The gRPC service depends on this abstraction
// rather than a concrete data store, making it easy to substitute
// alternate implementations and improving testability.
//
// All methods accept a context for cancellation and deadlines.  They
// return an error if the operation failed or a user was not found.
// When a user is not found, GetUser returns a nil *pb.User and a nil
// error.
type UserStore interface {
	// CreateUser stores a new user with the provided name and email.  It
	// returns the created user with an assigned unique ID.
	CreateUser(ctx context.Context, name, email string) (*pb.User, error)
	// GetUser returns the user identified by id or nil if the user
	// does not exist.  A nil error is returned when the user isn't found.
	GetUser(ctx context.Context, id int32) (*pb.User, error)
	// ListUsers returns all users in the store.  The returned slice
	// should be ordered consistently (e.g. insertion order) but no
	// ordering guarantee is required by this interface.
	ListUsers(ctx context.Context) ([]*pb.User, error)
}
