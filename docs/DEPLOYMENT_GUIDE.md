# Deployment Guide

## Overview

This guide covers deploying the 254Carbon ML Platform in different environments, from local development to production Kubernetes clusters.

## Prerequisites

### Software Requirements
- Docker and Docker Compose
- Kubernetes cluster (for production)
- Helm 3.x (for Kubernetes deployment)
- Python 3.9+ (for local development)
- Git

### Hardware Requirements
- **Development**: 8GB RAM, 4 CPU cores
- **Staging**: 16GB RAM, 8 CPU cores
- **Production**: 32GB RAM, 16 CPU cores, GPU nodes for embeddings

## Local Development Deployment

### Quick Start
```bash
# Clone repository
git clone <repository-url>
cd 254carbon-ml

# Install dependencies
make install

# Start all services
make docker-up

# Verify deployment
curl http://localhost:5000/health  # MLflow
curl http://localhost:9005/health  # Model Serving
curl http://localhost:9006/health  # Embedding Service
curl http://localhost:9007/health  # Search Service
```

### Service URLs
- **MLflow UI**: http://localhost:5000
- **Model Serving API**: http://localhost:9005
- **Embedding Service API**: http://localhost:9006
- **Search Service API**: http://localhost:9007
- **Jaeger UI**: http://localhost:16686
- **MinIO Console**: http://localhost:9001

### Training a Model
```bash
# Generate training data and train a model
cd training/curve_forecaster
python train.py --curve-type yield --model-type ensemble

# Promote model to production
python ../../scripts/promote_model.py --model curve_forecaster --stage Production
```

## Staging Deployment

### Infrastructure Setup
```bash
# Create staging namespace
kubectl create namespace ml-platform-staging

# Apply base configurations
kubectl apply -f k8s/base/ -n ml-platform-staging

# Deploy external dependencies
helm install postgres bitnami/postgresql \
  --set auth.postgresPassword=staging_password \
  --set primary.persistence.size=20Gi \
  -n ml-platform-staging

helm install redis bitnami/redis \
  --set auth.enabled=false \
  --set replica.replicaCount=1 \
  -n ml-platform-staging

helm install minio bitnami/minio \
  --set auth.rootUser=minioadmin \
  --set auth.rootPassword=staging_password \
  --set persistence.size=50Gi \
  -n ml-platform-staging
```

### Service Deployment
```bash
# Update image tags for staging
kubectl set image deployment/model-serving \
  model-serving=ghcr.io/254carbon/ml-platform-model-serving:staging \
  -n ml-platform-staging

kubectl set image deployment/embedding-service \
  embedding-service=ghcr.io/254carbon/ml-platform-embedding:staging \
  -n ml-platform-staging

kubectl set image deployment/search-service \
  search-service=ghcr.io/254carbon/ml-platform-search:staging \
  -n ml-platform-staging

# Wait for rollout
kubectl rollout status deployment/model-serving -n ml-platform-staging
kubectl rollout status deployment/embedding-service -n ml-platform-staging
kubectl rollout status deployment/search-service -n ml-platform-staging
```

### Verification
```bash
# Check pod status
kubectl get pods -n ml-platform-staging

# Check service health
kubectl port-forward svc/model-serving 9005:9005 -n ml-platform-staging &
curl http://localhost:9005/health

# Run integration tests
pytest tests/integration/ -v --staging
```

## Production Deployment

### Security Setup
```bash
# Setup Vault for secret management
cd security/
./setup-vault.sh

# Create TLS certificates
kubectl create secret tls ml-platform-tls \
  --cert=path/to/tls.crt \
  --key=path/to/tls.key \
  -n ml-platform-production
```

### High Availability Infrastructure
```bash
# Create production namespace
kubectl create namespace ml-platform-production

# Deploy PostgreSQL cluster
helm install postgres-ha bitnami/postgresql-ha \
  --set postgresql.replicaCount=3 \
  --set postgresql.persistence.size=100Gi \
  --set postgresql.postgresqlPassword=<secure-password> \
  -n ml-platform-production

# Deploy Redis cluster
helm install redis-cluster bitnami/redis-cluster \
  --set cluster.nodes=6 \
  --set persistence.size=20Gi \
  -n ml-platform-production

# Deploy MinIO cluster
helm install minio-ha bitnami/minio \
  --set mode=distributed \
  --set statefulset.replicaCount=4 \
  --set persistence.size=200Gi \
  --set auth.rootPassword=<secure-password> \
  -n ml-platform-production
```

### Service Deployment
```bash
# Apply production configurations
kubectl apply -f k8s/production/ -n ml-platform-production

# Deploy services with production images
kubectl set image deployment/model-serving \
  model-serving=ghcr.io/254carbon/ml-platform-model-serving:v1.0.0 \
  -n ml-platform-production

kubectl set image deployment/embedding-service \
  embedding-service=ghcr.io/254carbon/ml-platform-embedding:v1.0.0 \
  -n ml-platform-production

kubectl set image deployment/search-service \
  search-service=ghcr.io/254carbon/ml-platform-search:v1.0.0 \
  -n ml-platform-production

# Configure ingress
kubectl apply -f k8s/production/ingress.yaml -n ml-platform-production
```

### Monitoring Setup
```bash
# Deploy Prometheus
helm install prometheus prometheus-community/kube-prometheus-stack \
  --set prometheus.prometheusSpec.retention=30d \
  --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=100Gi \
  -n monitoring --create-namespace

# Deploy Grafana dashboards
kubectl create configmap ml-platform-dashboards \
  --from-file=monitoring/grafana/dashboards/ \
  -n monitoring

# Deploy alerting rules
kubectl create configmap ml-platform-alerts \
  --from-file=monitoring/prometheus/alerts.yml \
  -n monitoring
```

## Configuration Management

### Environment Variables
```bash
# Required environment variables for each service
export ML_ENV=production
export ML_VECTOR_DB_DSN="postgresql://user:pass@postgres:5432/mlflow"
export ML_REDIS_URL="redis://redis-cluster:6379"
export ML_MLFLOW_TRACKING_URI="http://mlflow-server:5000"
export ML_JWT_SECRET_KEY="<secure-secret-key>"
```

### Kubernetes Secrets
```bash
# Create secrets from Vault
kubectl create secret generic ml-platform-secrets \
  --from-literal=ML_VECTOR_DB_DSN="$(vault kv get -field=dsn secret/ml-platform/database/postgres)" \
  --from-literal=ML_JWT_SECRET_KEY="$(vault kv get -field=secret_key secret/ml-platform/jwt/main)" \
  -n ml-platform-production
```

### ConfigMaps
```bash
# Update configuration
kubectl create configmap ml-platform-config \
  --from-env-file=config/production.env \
  -n ml-platform-production
```

## Scaling Configuration

### Horizontal Pod Autoscaling
```yaml
# Model Serving HPA
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: model-serving-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: model-serving
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "100"
```

### Vertical Pod Autoscaling
```yaml
# Embedding Service VPA
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: embedding-service-vpa
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: embedding-service
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: embedding-service
      maxAllowed:
        cpu: 4
        memory: 8Gi
      minAllowed:
        cpu: 500m
        memory: 1Gi
```

## Backup and Recovery

### Database Backup
```bash
# Automated PostgreSQL backup
kubectl create cronjob postgres-backup \
  --image=postgres:16 \
  --schedule="0 2 * * *" \
  --restart=OnFailure \
  -- /bin/bash -c "pg_dump -h postgres -U mlflow mlflow | gzip > /backup/mlflow-$(date +%Y%m%d).sql.gz"
```

### Model Artifact Backup
```bash
# MinIO backup to cloud storage
kubectl create cronjob minio-backup \
  --image=minio/mc \
  --schedule="0 3 * * *" \
  --restart=OnFailure \
  -- /bin/bash -c "mc mirror minio/mlflow-artifacts s3://backup-bucket/mlflow-artifacts/"
```

### Disaster Recovery
```bash
# Restore from backup
kubectl create job postgres-restore \
  --image=postgres:16 \
  -- /bin/bash -c "gunzip -c /backup/mlflow-20240101.sql.gz | psql -h postgres -U mlflow mlflow"
```

## Health Checks and Monitoring

### Service Health Endpoints
- **Model Serving**: `GET /health`
- **Embedding Service**: `GET /health`
- **Search Service**: `GET /health`
- **MLflow Server**: `GET /health`

### Metrics Endpoints
- **All Services**: `GET /metrics` (Prometheus format)

### Readiness Probes
```yaml
readinessProbe:
  httpGet:
    path: /health
    port: 9005
  initialDelaySeconds: 10
  periodSeconds: 5
  timeoutSeconds: 3
  failureThreshold: 3
```

### Liveness Probes
```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 9005
  initialDelaySeconds: 30
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 5
```

## Performance Tuning

### Database Optimization
```sql
-- PostgreSQL configuration for production
ALTER SYSTEM SET shared_buffers = '4GB';
ALTER SYSTEM SET effective_cache_size = '12GB';
ALTER SYSTEM SET maintenance_work_mem = '1GB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;
SELECT pg_reload_conf();
```

### Redis Optimization
```bash
# Redis configuration for production
redis-cli CONFIG SET maxmemory 4gb
redis-cli CONFIG SET maxmemory-policy allkeys-lru
redis-cli CONFIG SET save "900 1 300 10 60 10000"
```

### JVM Tuning (if using Java components)
```bash
export JAVA_OPTS="-Xms2g -Xmx4g -XX:+UseG1GC -XX:MaxGCPauseMillis=200"
```

## Security Hardening

### Network Security
```yaml
# Network Policy
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: ml-platform-network-policy
spec:
  podSelector:
    matchLabels:
      app: ml-platform
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ml-platform
    ports:
    - protocol: TCP
      port: 9005
    - protocol: TCP
      port: 9006
    - protocol: TCP
      port: 9007
```

### Pod Security Standards
```yaml
apiVersion: v1
kind: Pod
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    fsGroup: 2000
  containers:
  - name: model-serving
    securityContext:
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      capabilities:
        drop:
        - ALL
```

### RBAC Configuration
```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: ml-platform-role
rules:
- apiGroups: [""]
  resources: ["configmaps", "secrets"]
  verbs: ["get", "list"]
- apiGroups: ["apps"]
  resources: ["deployments"]
  verbs: ["get", "list", "update"]
```

## Troubleshooting

### Common Issues

#### Service Won't Start
```bash
# Check pod logs
kubectl logs -f deployment/model-serving -n ml-platform

# Check events
kubectl get events --sort-by=.metadata.creationTimestamp -n ml-platform

# Check resource constraints
kubectl describe pod <pod-name> -n ml-platform
```

#### Database Connection Issues
```bash
# Test database connectivity
kubectl run postgres-test --image=postgres:16 --rm -it --restart=Never \
  -- psql -h postgres -U mlflow -d mlflow -c "SELECT 1;"

# Check database logs
kubectl logs -f statefulset/postgres -n ml-platform
```

#### Performance Issues
```bash
# Check resource usage
kubectl top pods -n ml-platform

# Check HPA status
kubectl get hpa -n ml-platform

# Check metrics
kubectl port-forward svc/model-serving 9005:9005 -n ml-platform &
curl http://localhost:9005/metrics
```

### Debugging Commands
```bash
# Get all resources
kubectl get all -n ml-platform

# Describe problematic pod
kubectl describe pod <pod-name> -n ml-platform

# Execute commands in pod
kubectl exec -it deployment/model-serving -- /bin/bash

# Check service endpoints
kubectl get endpoints -n ml-platform

# View recent events
kubectl get events --sort-by=.metadata.creationTimestamp -n ml-platform | tail -20
```

## Maintenance Tasks

### Regular Maintenance
```bash
# Update dependencies (monthly)
make update-deps

# Rotate secrets (quarterly)
./security/rotate-secrets.sh

# Clean up old model artifacts (weekly)
python scripts/cleanup_old_artifacts.py --days 30

# Database maintenance (weekly)
kubectl exec -it statefulset/postgres -- psql -U mlflow -d mlflow -c "VACUUM ANALYZE;"
```

### Model Updates
```bash
# Deploy new model version
python scripts/promote_model.py --model curve_forecaster_v2 --stage Production

# Rollback if needed
python scripts/promote_model.py --model curve_forecaster --stage Production
```

### Scaling Operations
```bash
# Manual scaling
kubectl scale deployment model-serving --replicas=10 -n ml-platform

# Update resource limits
kubectl patch deployment model-serving \
  -p '{"spec":{"template":{"spec":{"containers":[{"name":"model-serving","resources":{"limits":{"memory":"4Gi","cpu":"2000m"}}}]}}}}' \
  -n ml-platform
```

## Monitoring and Alerting

### Key Metrics to Monitor
- **Latency**: P95 response times for all endpoints
- **Throughput**: Requests per second
- **Error Rate**: 4xx and 5xx error rates
- **Resource Usage**: CPU, memory, GPU utilization
- **Model Performance**: Prediction accuracy, drift detection

### Alert Thresholds
- **Critical**: P95 latency > 1s, Error rate > 10%, Service down
- **Warning**: P95 latency > 500ms, Error rate > 5%, High resource usage

### Dashboard URLs
- **Grafana**: http://grafana.254carbon.local
- **Prometheus**: http://prometheus.254carbon.local
- **Jaeger**: http://jaeger.254carbon.local
- **AlertManager**: http://alertmanager.254carbon.local

## Disaster Recovery

### Backup Strategy
- **Database**: Daily automated backups with 30-day retention
- **Model Artifacts**: Continuous replication to cloud storage
- **Configuration**: Version controlled in Git
- **Secrets**: Encrypted backup in secure storage

### Recovery Procedures
1. **Service Failure**: Automatic restart via Kubernetes
2. **Node Failure**: Automatic pod rescheduling
3. **Database Failure**: Restore from latest backup
4. **Complete Cluster Failure**: Restore to new cluster from backups

### RTO/RPO Targets
- **Recovery Time Objective (RTO)**: < 1 hour
- **Recovery Point Objective (RPO)**: < 15 minutes
- **Data Loss**: Zero tolerance for model artifacts and training data

## Performance Optimization

### Database Tuning
- Connection pooling: 20 connections per service
- Query optimization: Proper indexing on frequently queried columns
- Partitioning: Time-based partitioning for large tables
- Read replicas: For read-heavy workloads

### Caching Strategy
- **Redis**: Query result caching, embedding caching
- **Application**: In-memory model caching
- **CDN**: Static asset caching (future)

### Resource Allocation
- **CPU**: 2-4 cores per service instance
- **Memory**: 2-8GB per service instance
- **GPU**: Dedicated GPU nodes for embedding service
- **Storage**: SSD storage for databases, object storage for artifacts

## Security Best Practices

### Authentication
- JWT tokens for API authentication
- Service-to-service authentication with mutual TLS
- API key rotation every 90 days
- Strong password policies

### Authorization
- Role-based access control (RBAC)
- Principle of least privilege
- Regular access reviews
- Audit logging for sensitive operations

### Data Protection
- Encryption at rest for all sensitive data
- Encryption in transit (TLS 1.3)
- Data masking in logs and metrics
- Secure key management with Vault

### Network Security
- Network policies to restrict pod-to-pod communication
- Private subnets for database and cache layers
- WAF (Web Application Firewall) for external APIs
- DDoS protection

## Compliance and Auditing

### Audit Logging
- All API requests logged with user context
- Model promotion and deployment events
- Data access and modification events
- Security events and authentication failures

### Compliance Requirements
- Data retention policies
- Access control documentation
- Security scanning reports
- Vulnerability management

### Regular Security Tasks
- Weekly vulnerability scans
- Monthly security reviews
- Quarterly penetration testing
- Annual security audits
