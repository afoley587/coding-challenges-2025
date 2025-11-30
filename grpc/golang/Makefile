
build:
	go build ./...

docker:
	docker compose up -d --build

fmt:
	go fmt

proto:
	protoc --go_out=. --go_opt=paths=source_relative \
		--go-grpc_out=. --go-grpc_opt=paths=source_relative \
		proto/users.proto

run-server:
	go run . server run \
		--addr 0.0.0.0:9090 \
		--redis-address 127.0.0.1:6379

test:
	go test ./...