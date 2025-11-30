package server

import (
	"context"
	"fmt"
	"log"
	"encoding/json"
	"github.com/redis/go-redis/v9/maintnotifications"

	pb "github.com/afoley587/coding-challenges-2025/grpc-golang-api/proto"
	redis "github.com/redis/go-redis/v9"
	codes "google.golang.org/grpc/codes"
	status "google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/emptypb"
)

func NewServer(redisAddr string) *GrpcGolangApiServer {
	rdb := redis.NewClient(&redis.Options{
		Addr:     redisAddr,
		Password: "", // no password set
		DB:       0,  // use default DB
		MaintNotificationsConfig: &maintnotifications.Config{
			Mode: maintnotifications.ModeDisabled,
		},
	})
	s := &GrpcGolangApiServer{r: rdb}
	log.Println("Started")
	return s
}

func (s *GrpcGolangApiServer) ListUsers(_ *emptypb.Empty, stream pb.GrpcGolangAPI_ListUsersServer) error {
	log.Println("In ListUsers")

	var usersJson map[int32]*pb.User

	users, err := s.r.Get(context.Background(), "users").Result()

	fmt.Println(users)

	if err == redis.Nil {
		log.Println("No redis users yet.")
		return nil
	} else if err != nil {
		log.Fatalf("failed to fetch from redis: %v", err)
		return status.Errorf(codes.Unimplemented, "failed to fetch from redis")
	}

	err = json.Unmarshal([]byte(users), &usersJson)

	if err != nil {
		log.Fatalf("failed to unmarshal users: %v", err)
		return status.Errorf(codes.Unimplemented, "failed to unmarshal users")
	}

	for _, u := range usersJson {
		fmt.Printf("Sending %v\n", u)
		if err := stream.Send(u); err != nil {
			return err
		}
	}
	return nil
}

func (s *GrpcGolangApiServer) CreateUser(ctx context.Context, r *pb.CreateUserRequest) (*pb.User, error) {
	log.Println("In CreateUser")

	var usersJson map[int32]*pb.User

	users, err := s.r.Get(context.Background(), "users").Result()

	if err == redis.Nil {
		log.Println("No redis users yet.")
		users = `{}`
	} else if err != nil {
		log.Fatalf("failed to fetch from redis: %v", err)
		return nil, status.Errorf(codes.Unimplemented, "failed to fetch from redis")
	}

	err = json.Unmarshal([]byte(users), &usersJson)

	if err != nil {
		log.Fatalf("failed to unmarshal users: %v", err)
		return nil, status.Errorf(codes.Unimplemented, "failed to unmarshal users")
	}

	var nextId int32
	nextId = int32(len(usersJson))
	u := &pb.User{
		Id:    &nextId,
		Name:  r.Name,
		Email: r.Email,
	}

	usersJson[nextId] = u

	jsonBytes, err := json.Marshal(usersJson)

	err = s.r.Set(ctx, "users", jsonBytes, 0).Err()
	if err != nil {
		log.Fatalf("failed to save to redis: %v", err)
		return nil, status.Errorf(codes.Unimplemented, "failed to save to redis")
	}

	return u, nil
}

func (s *GrpcGolangApiServer) GetUser(ctx context.Context, r *pb.GetUserRequest) (*pb.User, error) {
	var usersJson map[int32]*pb.User
	users, err := s.r.Get(ctx, "users").Result()

	if err != nil {
		log.Fatalf("failed to fetch from redis: %v", err)
		return nil, status.Errorf(codes.Unimplemented, "failed to fetch from redis")
	}

	err = json.Unmarshal([]byte(users), &usersJson)

	if err != nil {
		log.Fatalf("failed to unmarshal users: %v", err)
		return nil, status.Errorf(codes.Unimplemented, "failed to unmarshal users")
	}

	if user, ok := usersJson[*r.Id]; ok {
		return user, nil
	}

	return nil, nil
}
