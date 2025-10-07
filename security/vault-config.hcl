# Vault configuration for ML platform secrets

# Storage backend
storage "file" {
  path = "/vault/data"
}

# Listener configuration
listener "tcp" {
  address = "0.0.0.0:8200"
  tls_disable = 1  # Disable for development, enable in production
}

# API address
api_addr = "http://127.0.0.1:8200"

# Cluster address
cluster_addr = "https://127.0.0.1:8201"

# UI
ui = true

# Logging
log_level = "INFO"

# Disable mlock for development
disable_mlock = true

# Default lease TTL
default_lease_ttl = "768h"
max_lease_ttl = "8760h"
