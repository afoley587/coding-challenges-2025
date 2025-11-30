package server

import (
    "context"

    pb "github.com/afoley587/coding-challenges-2025/grpc-golang-api/proto"
    "github.com/afoley587/coding-challenges-2025/grpc-golang-api/internal/store"
    "google.golang.org/grpc/codes"
    "google.golang.org/grpc/status"
    "google.golang.org/protobuf/types/known/emptypb"
)

// grpcServer implements the generated gRPC interface by delegating
// operations to a UserStore.  It contains no storage logic of its own.
//
// The store field is exported only through the interface so that
// external callers cannot manipulate it directly.  Use NewGRPCServer
// to construct an instance.
type grpcServer struct {
    pb.UnimplementedGrpcGolangAPIServer
    store store.UserStore
}

// NewGRPCServer constructs a gRPC service implementation backed by the
// provided store.  The returned value implements
// pb.GrpcGolangAPIServer.
func NewGRPCServer(store store.UserStore) pb.GrpcGolangAPIServer {
    return &grpcServer{store: store}
}

// GetUser returns a single user identified by id.  If the user is not
// found, a NotFound status code is returned.
func (s *grpcServer) GetUser(ctx context.Context, req *pb.GetUserRequest) (*pb.User, error) {
    if req == nil || req.Id == nil {
        return nil, status.Error(codes.InvalidArgument, "id is required")
    }
    user, err := s.store.GetUser(ctx, req.GetId())
    if err != nil {
        return nil, status.Errorf(codes.Internal, "get user failed: %v", err)
    }
    if user == nil {
        return nil, status.Error(codes.NotFound, "user not found")
    }
    return user, nil
}

// ListUsers streams all users to the client.  If no users exist, the
// stream is closed without sending any messages.
func (s *grpcServer) ListUsers(_ *emptypb.Empty, stream pb.GrpcGolangAPI_ListUsersServer) error {
    users, err := s.store.ListUsers(stream.Context())
    if err != nil {
        return status.Errorf(codes.Internal, "list users failed: %v", err)
    }
    for _, u := range users {
        if err := stream.Send(u); err != nil {
            return err
        }
    }
    return nil
}

// CreateUser creates a new user with the provided name and email.  Name
// and email are required; if either is empty, an InvalidArgument
// status is returned.
func (s *grpcServer) CreateUser(ctx context.Context, req *pb.CreateUserRequest) (*pb.User, error) {
    if req == nil || req.Name == nil || req.Email == nil {
        return nil, status.Error(codes.InvalidArgument, "name and email are required")
    }
    user, err := s.store.CreateUser(ctx, req.GetName(), req.GetEmail())
    if err != nil {
        return nil, status.Errorf(codes.Internal, "create user failed: %v", err)
    }
    return user, nil
}