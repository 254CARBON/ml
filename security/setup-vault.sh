#!/bin/bash
## Setup Vault (development) for ML Platform secrets
# Purpose: Stand up a local Vault dev instance and populate baseline secrets
# DANGER: Dev mode is NOT secure. For production, use HA, TLS, proper unseal
#         keys and remove plaintext outputs.
# Prereqs: HashiCorp Vault CLI installed; openssl; Python with cryptography
# Usage: ./security/setup-vault.sh

set -e

echo "Setting up Vault for ML platform..."

# Start Vault server (development mode)
vault server -config=vault-config.hcl &
VAULT_PID=$!

# Wait for Vault to start
sleep 5

# Set Vault address
export VAULT_ADDR='http://127.0.0.1:8200'

# Initialize Vault (development mode)
vault operator init -key-shares=1 -key-threshold=1 > vault-keys.txt

# Extract unseal key and root token
UNSEAL_KEY=$(grep 'Unseal Key 1:' vault-keys.txt | awk '{print $NF}')
ROOT_TOKEN=$(grep 'Initial Root Token:' vault-keys.txt | awk '{print $NF}')

# Unseal Vault
vault operator unseal $UNSEAL_KEY

# Login with root token
vault auth $ROOT_TOKEN

# Enable KV secrets engine
vault secrets enable -path=secret kv-v2

# Create policies
vault policy write ml-platform-admin vault-policies.hcl

# Create service-specific policies
vault policy write model-serving-policy - <<EOF
path "secret/data/ml-platform/model-serving/*" {
  capabilities = ["read"]
}
path "secret/data/ml-platform/shared/*" {
  capabilities = ["read"]
}
EOF

vault policy write embedding-service-policy - <<EOF
path "secret/data/ml-platform/embedding-service/*" {
  capabilities = ["read"]
}
path "secret/data/ml-platform/shared/*" {
  capabilities = ["read"]
}
EOF

vault policy write search-service-policy - <<EOF
path "secret/data/ml-platform/search-service/*" {
  capabilities = ["read"]
}
path "secret/data/ml-platform/shared/*" {
  capabilities = ["read"]
}
EOF

# Store initial secrets
echo "Storing initial secrets..."

# Database credentials
vault kv put secret/ml-platform/database/postgres \
  username=mlflow \
  password=mlflow_password \
  host=postgres \
  port=5432 \
  database=mlflow

# MinIO credentials
vault kv put secret/ml-platform/shared/minio \
  access_key=minioadmin \
  secret_key=minioadmin123 \
  endpoint=http://minio:9000

# Redis URL
vault kv put secret/ml-platform/shared/redis \
  url=redis://redis:6379

# JWT secrets
vault kv put secret/ml-platform/shared/jwt \
  secret_key=$(openssl rand -base64 32) \
  algorithm=HS256

# Encryption keys
vault kv put secret/ml-platform/shared/encryption \
  key=$(python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")

# API keys for services
vault kv put secret/ml-platform/api-keys/model-serving \
  api_key=$(openssl rand -base64 32)

vault kv put secret/ml-platform/api-keys/embedding-service \
  api_key=$(openssl rand -base64 32)

vault kv put secret/ml-platform/api-keys/search-service \
  api_key=$(openssl rand -base64 32)

# Create service tokens
echo "Creating service tokens..."

# Model serving token
MODEL_SERVING_TOKEN=$(vault write -field=token auth/token/create \
  policies=model-serving-policy \
  ttl=8760h)

# Embedding service token
EMBEDDING_TOKEN=$(vault write -field=token auth/token/create \
  policies=embedding-service-policy \
  ttl=8760h)

# Search service token
SEARCH_TOKEN=$(vault write -field=token auth/token/create \
  policies=search-service-policy \
  ttl=8760h)

# Save tokens to file
cat > service-tokens.txt <<EOF
Model Serving Token: $MODEL_SERVING_TOKEN
Embedding Service Token: $EMBEDDING_TOKEN
Search Service Token: $SEARCH_TOKEN
EOF

echo "Vault setup completed!"
echo "Root token: $ROOT_TOKEN"
echo "Unseal key: $UNSEAL_KEY"
echo "Service tokens saved to service-tokens.txt"
echo ""
echo "To use Vault:"
echo "export VAULT_ADDR='http://127.0.0.1:8200'"
echo "export VAULT_TOKEN='$ROOT_TOKEN'"
echo ""
echo "WARNING: This is a development setup. For production:"
echo "1. Use proper TLS certificates"
echo "2. Use multiple unseal keys"
echo "3. Store keys securely"
echo "4. Enable audit logging"
echo "5. Configure proper authentication methods"
