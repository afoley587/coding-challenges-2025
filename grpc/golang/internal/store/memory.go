package store

import (
	"context"
	"sync"

	pb "github.com/afoley587/coding-challenges-2025/grpc-golang-api/proto"
)

// InMemoryStore is an implementation of UserStore backed by a simple
// in‑memory map.  It is safe for concurrent use and intended primarily
// for unit tests and development.  Data stored in this store is not
// persisted beyond the lifetime of the process.
type InMemoryStore struct {
	mu    sync.Mutex
	users map[int32]*pb.User
}

// NewInMemoryStore constructs an empty in‑memory store.
func NewInMemoryStore() *InMemoryStore {
	return &InMemoryStore{
		users: make(map[int32]*pb.User),
	}
}

// CreateUser inserts a new user into the store and returns the created
// entity with an assigned identifier.  IDs are assigned sequentially
// starting from 0.  This method is safe for concurrent use.
func (s *InMemoryStore) CreateUser(ctx context.Context, name, email string) (*pb.User, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	id := int32(len(s.users))
	// Copy values into separate variables so that pointers refer to
	// stable memory rather than the stack variables.
	n := name
	e := email
	user := &pb.User{
		Id:    &id,
		Name:  &n,
		Email: &e,
	}
	s.users[id] = user
	return user, nil
}

// GetUser retrieves a user by id.  It returns (nil, nil) if the user
// does not exist.  This method is safe for concurrent use.
func (s *InMemoryStore) GetUser(ctx context.Context, id int32) (*pb.User, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if u, ok := s.users[id]; ok {
		return u, nil
	}
	return nil, nil
}

// ListUsers returns all users stored.  This method is safe for
// concurrent use.  It copies pointers from the underlying map into
// a slice.  The order corresponds to the insertion order but this
// implementation does not guarantee a particular ordering.
func (s *InMemoryStore) ListUsers(ctx context.Context) ([]*pb.User, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	users := make([]*pb.User, 0, len(s.users))
	for _, u := range s.users {
		users = append(users, u)
	}
	return users, nil
}
