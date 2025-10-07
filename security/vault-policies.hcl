# Vault policies for ML platform services

# ML Platform Admin Policy
path "secret/ml-platform/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

path "secret/ml-platform/admin/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

# Model Serving Service Policy
path "secret/ml-platform/model-serving/*" {
  capabilities = ["read"]
}

path "secret/ml-platform/shared/*" {
  capabilities = ["read"]
}

# Embedding Service Policy
path "secret/ml-platform/embedding-service/*" {
  capabilities = ["read"]
}

path "secret/ml-platform/shared/*" {
  capabilities = ["read"]
}

# Search Service Policy
path "secret/ml-platform/search-service/*" {
  capabilities = ["read"]
}

path "secret/ml-platform/shared/*" {
  capabilities = ["read"]
}

# MLflow Server Policy
path "secret/ml-platform/mlflow-server/*" {
  capabilities = ["read"]
}

path "secret/ml-platform/shared/*" {
  capabilities = ["read"]
}

# Database credentials (read-only for services)
path "secret/ml-platform/database/*" {
  capabilities = ["read"]
}

# API keys (read-only for services)
path "secret/ml-platform/api-keys/*" {
  capabilities = ["read"]
}

# Encryption keys (read-only for services)
path "secret/ml-platform/encryption/*" {
  capabilities = ["read"]
}

# JWT secrets (read-only for services)
path "secret/ml-platform/jwt/*" {
  capabilities = ["read"]
}
