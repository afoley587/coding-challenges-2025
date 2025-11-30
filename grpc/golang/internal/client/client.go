package client

import (
	"context"
	"fmt"
	"io"
	"log"
	"time"

	pb "github.com/afoley587/coding-challenges-2025/grpc-golang-api/proto"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/emptypb"
)


func ListUsers(serverAddr string) {

	var opts []grpc.DialOption
	opts = append(opts, grpc.WithTransportCredentials(insecure.NewCredentials()))

	conn, err := grpc.NewClient(serverAddr, opts...)
	if err != nil {
		log.Fatalf("fail to dial: %v", err)
	}
	defer conn.Close()
	client := pb.NewGrpcGolangAPIClient(conn)

	ctx, cxl := context.WithTimeout(context.Background(), 10*time.Second)
	defer cxl()

	stream, err := client.ListUsers(ctx, &emptypb.Empty{})

	if err != nil {
		log.Fatalf("client.ListUsers failed: %v", err)
	}
	for {
		user, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Fatalf("client.ListUsers failed: %v", err)
		}
		log.Printf("%v\n", user)
	}
}

func CreateUser(serverAddr string) {

	var opts []grpc.DialOption
	opts = append(opts, grpc.WithTransportCredentials(insecure.NewCredentials()))

	conn, err := grpc.NewClient(serverAddr, opts...)
	if err != nil {
		log.Fatalf("fail to dial: %v", err)
	}
	defer conn.Close()
	client := pb.NewGrpcGolangAPIClient(conn)

	ctx, cxl := context.WithTimeout(context.Background(), 10*time.Second)
	defer cxl()

	newUser := &pb.CreateUserRequest{
		Name:  proto.String("Alex"),
		Email: proto.String("Alex"),
	}

	user, err := client.CreateUser(ctx, newUser)

	if err != nil {
		log.Fatalf("client.CreateUser failed: %v", err)
	}

	fmt.Printf("User created: %v\n", user)
}
