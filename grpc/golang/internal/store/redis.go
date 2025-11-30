package store

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"fmt"

	pb "github.com/afoley587/coding-challenges-2025/grpc-golang-api/proto"
	redis "github.com/redis/go-redis/v9"
	"github.com/redis/go-redis/v9/maintnotifications"
)

// RedisStore is an implementation of UserStore backed by a Redis key.  It
// stores all users in a single JSON document stored at the "users"
// key.  While this implementation is straightforward, it is not
// optimized for high concurrency or large data sets; consider using
// per‑user keys or hashes for production systems.  All operations are
// performed atomically on the JSON blob, replacing the entire value on
// updates.
type RedisStore struct {
	client *redis.Client
}

// NewRedisStore connects to a Redis instance at the provided address and
// returns a store using the "users" key.  A ping is performed to
// verify connectivity.
func NewRedisStore(addr string, password string, tls *tls.Config) (*RedisStore, error) {
	opts := &redis.Options{
		Addr:     addr,
		Password: password, // empty string means no auth
		DB:       0,
		MaintNotificationsConfig: &maintnotifications.Config{
			Mode: maintnotifications.ModeDisabled,
		},
	}
	if tls != nil {
		opts.TLSConfig = tls
	}

	client := redis.NewClient(opts)
	if err := client.Ping(context.Background()).Err(); err != nil {
		return nil, fmt.Errorf("redis ping failed: %w", err)
	}
	return &RedisStore{client: client}, nil
}

// CreateUser inserts a new user into Redis.  The entire set of users is
// fetched, updated, and written back as one operation.  If the
// underlying JSON blob cannot be decoded or written, an error is
// returned.
func (s *RedisStore) CreateUser(ctx context.Context, name, email string) (*pb.User, error) {
	users, err := s.fetchAll(ctx)
	if err != nil {
		return nil, err
	}
	id := int32(len(users))
	n := name
	e := email
	user := &pb.User{
		Id:    &id,
		Name:  &n,
		Email: &e,
	}
	users[id] = user
	if err := s.saveAll(ctx, users); err != nil {
		return nil, err
	}
	return user, nil
}

// GetUser retrieves a user by id from Redis.  It returns (nil, nil) if
// the user does not exist.
func (s *RedisStore) GetUser(ctx context.Context, id int32) (*pb.User, error) {
	users, err := s.fetchAll(ctx)
	if err != nil {
		return nil, err
	}
	if u, ok := users[id]; ok {
		return u, nil
	}
	return nil, nil
}

// ListUsers returns all users stored in Redis.  On a fresh store or
// when the key doesn't exist, an empty slice and nil error are
// returned.
func (s *RedisStore) ListUsers(ctx context.Context) ([]*pb.User, error) {
	users, err := s.fetchAll(ctx)
	if err != nil {
		return nil, err
	}
	result := make([]*pb.User, 0, len(users))
	for _, u := range users {
		result = append(result, u)
	}
	return result, nil
}

// fetchAll reads the JSON blob from Redis and decodes it into a map.  If
// the key does not exist, an empty map is returned.  On any other
// error, an error is returned.
func (s *RedisStore) fetchAll(ctx context.Context) (map[int32]*pb.User, error) {
	usersJSON, err := s.client.Get(ctx, "users").Result()
	if err == redis.Nil {
		return make(map[int32]*pb.User), nil
	}
	if err != nil {
		return nil, fmt.Errorf("redis get failed: %w", err)
	}
	var users map[int32]*pb.User
	if err := json.Unmarshal([]byte(usersJSON), &users); err != nil {
		return nil, fmt.Errorf("failed to unmarshal users json: %w", err)
	}
	if users == nil {
		users = make(map[int32]*pb.User)
	}
	return users, nil
}

// saveAll writes the provided map of users to Redis as a JSON blob.  If
// marshaling fails or the write fails, an error is returned.
func (s *RedisStore) saveAll(ctx context.Context, users map[int32]*pb.User) error {
	data, err := json.Marshal(users)
	if err != nil {
		return fmt.Errorf("failed to marshal users: %w", err)
	}
	if err := s.client.Set(ctx, "users", data, 0).Err(); err != nil {
		return fmt.Errorf("redis set failed: %w", err)
	}
	return nil
}
