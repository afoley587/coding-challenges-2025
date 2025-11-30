package server

import (
	pb "github.com/afoley587/coding-challenges-2025/grpc-golang-api/proto"
	redis "github.com/redis/go-redis/v9"
)

type GrpcGolangApiServer struct {
	pb.UnimplementedGrpcGolangAPIServer
	r *redis.Client
}

