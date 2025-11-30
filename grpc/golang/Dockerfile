FROM --platform=$BUILDPLATFORM golang:1.24-alpine AS builder

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY . .

ARG TARGETOS TARGETARCH
RUN CGO_ENABLED=0 GOOS="$TARGETOS" GOARCH="$TARGETARCH" go build -o /out/app .

FROM scratch

COPY --from=builder /out/app /app

ENTRYPOINT ["/app"]