
build:
	go build

fmt:
	go fmt

proto:
	protoc --go_out=. --go_opt=paths=source_relative \
		--go-grpc_out=. --go-grpc_opt=paths=source_relative \
		proto/users.proto

test:
	go test ./...