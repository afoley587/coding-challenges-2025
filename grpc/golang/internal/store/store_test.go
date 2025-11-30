package store_test

import (
	"context"
	"testing"

	"github.com/afoley587/coding-challenges-2025/grpc-golang-api/internal/store"
)

// TestInMemoryStoreCRUD exercises the basic Create/Get/List behavior of the
// in‑memory store.  It ensures IDs are assigned sequentially, data is
// persisted, and nonexistent lookups return nil without error.
func TestInMemoryStoreCRUD(t *testing.T) {
	s := store.NewInMemoryStore()
	ctx := context.Background()

	// Create a user and verify its contents.
	user, err := s.CreateUser(ctx, "Alice", "alice@example.com")
	if err != nil {
		t.Fatalf("CreateUser returned error: %v", err)
	}
	if user.GetId() != 0 {
		t.Fatalf("expected id 0, got %d", user.GetId())
	}
	if got, want := user.GetName(), "Alice"; got != want {
		t.Fatalf("expected name %q, got %q", want, got)
	}
	if got, want := user.GetEmail(), "alice@example.com"; got != want {
		t.Fatalf("expected email %q, got %q", want, got)
	}

	// Retrieve the same user by ID.
	got, err := s.GetUser(ctx, 0)
	if err != nil {
		t.Fatalf("GetUser returned error: %v", err)
	}
	if got == nil || got.GetName() != "Alice" {
		t.Fatalf("unexpected user retrieved: %+v", got)
	}

	// List users should return exactly one entry.
	users, err := s.ListUsers(ctx)
	if err != nil {
		t.Fatalf("ListUsers returned error: %v", err)
	}
	if len(users) != 1 {
		t.Fatalf("expected 1 user, got %d", len(users))
	}
	if users[0].GetName() != "Alice" {
		t.Fatalf("expected first user to be Alice, got %s", users[0].GetName())
	}

	// Requesting a nonexistent user should return nil without error.
	none, err := s.GetUser(ctx, 99)
	if err != nil {
		t.Fatalf("GetUser for nonexistent id returned error: %v", err)
	}
	if none != nil {
		t.Fatalf("expected nil for nonexistent user, got %+v", none)
	}
}
