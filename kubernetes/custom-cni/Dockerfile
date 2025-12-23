# Build the custom bridge CNI plugin
FROM golang:1.25 AS builder
WORKDIR /workspace
COPY go.mod ./
COPY go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux GOARCH=${TARGETARCH:-amd64} go build -o /out/custom-bridge

FROM alpine:3.23
RUN mkdir -p /cni/bin /config
COPY --from=builder /out/custom-bridge /cni/bin/custom-bridge
COPY k8s/00-customcni.conflist /config/00-customcni.conflist
ENTRYPOINT ["/bin/sh", "-c", "cp /cni/bin/custom-bridge /host/opt/cni/bin/ && cp /config/00-customcni.conflist /host/etc/cni/net.d/00-customcni.conflist && sleep infinity"]
