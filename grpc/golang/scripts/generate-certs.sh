#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CERT_DIR="$ROOT_DIR/certs"

echo "==> Creating certificate directories..."
mkdir -p "$CERT_DIR/ca" "$CERT_DIR/server" "$CERT_DIR/client"


echo "==> Generating CA private key..."
openssl genrsa -out "$CERT_DIR/ca/ca.key" 4096

echo "==> Generating CA certificate..."
openssl req -x509 -new -nodes \
  -key "$CERT_DIR/ca/ca.key" \
  -sha256 -days 3650 \
  -out "$CERT_DIR/ca/ca.crt" \
  -subj "/C=US/ST=TX/L=Austin/O=MyOrg/OU=Dev/CN=MyLocalCA"


echo "==> Generating server private key..."
openssl genrsa -out "$CERT_DIR/server/server.key" 4096

echo "==> Generating server CSR..."
openssl req -new \
  -key "$CERT_DIR/server/server.key" \
  -out "$CERT_DIR/server/server.csr" \
  -subj "/C=US/ST=TX/L=Austin/O=MyOrg/OU=Dev/CN=localhost" # pragma: allowlist secret

echo "==> Creating server certificate extension file..."
cat > "$CERT_DIR/server/server.ext" <<EOF
authorityKeyIdentifier=keyid,issuer
basicConstraints=CA:FALSE
keyUsage = digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth
subjectAltName = @alt_names

[alt_names]
DNS.1 = localhost
DNS.2 = 127.0.0.1
EOF

echo "==> Signing server certificate with CA..."
openssl x509 -req \
  -in "$CERT_DIR/server/server.csr" \
  -CA "$CERT_DIR/ca/ca.crt" \
  -CAkey "$CERT_DIR/ca/ca.key" \
  -CAcreateserial \
  -out "$CERT_DIR/server/server.crt" \
  -days 365 \
  -sha256 \
  -extfile "$CERT_DIR/server/server.ext"

echo "==> Generating client private key..."
openssl genrsa -out "$CERT_DIR/client/client.key" 4096

echo "==> Generating client CSR..."
openssl req -new \
  -key "$CERT_DIR/client/client.key" \
  -out "$CERT_DIR/client/client.csr" \
  -subj "/C=US/ST=TX/L=Austin/O=MyOrg/OU=Dev/CN=grpc-client"

echo "==> Creating client certificate extension file..."
cat > "$CERT_DIR/client/client.ext" <<EOF
authorityKeyIdentifier=keyid,issuer
basicConstraints=CA:FALSE
keyUsage = digitalSignature
extendedKeyUsage = clientAuth
EOF

echo "==> Signing client certificate with CA..."
openssl x509 -req \
  -in "$CERT_DIR/client/client.csr" \
  -CA "$CERT_DIR/ca/ca.crt" \
  -CAkey "$CERT_DIR/ca/ca.key" \
  -CAcreateserial \
  -out "$CERT_DIR/client/client.crt" \
  -days 365 \
  -sha256 \
  -extfile "$CERT_DIR/client/client.ext"


###############################
# DONE
###############################
echo ""
echo "==> Certificates generated successfully!"
echo "    CA:      certs/ca/ca.crt"
echo "    SERVER:  certs/server/server.{key,crt}"
echo "    CLIENT:  certs/client/client.{key,crt}"
echo ""
echo "Use with mTLS:"
echo "    Server: --cert certs/server/server.crt --key certs/server/server.key --ca certs/ca/ca.crt"
echo "    Client: --tls-cert certs/client/client.crt --tls-key certs/client/client.key --tls-ca certs/ca/ca.crt"
