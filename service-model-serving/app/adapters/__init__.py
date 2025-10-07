"""Model adapters for input/output normalization.

Adapters translate between external API payloads and the model’s native
features/outputs. Keeping these translations in one place simplifies model
swaps and makes API contracts predictable.

Typical responsibilities
- Validate/transform request JSON into model feature tensors
- Convert model outputs into response schemas with metadata
"""
