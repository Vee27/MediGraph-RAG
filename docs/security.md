# Security

## Research Use Only

This project is intended for research and educational purposes.

It is not FDA-approved medical software and should not be used for clinical decision-making.

## Data Privacy

By default:

- ChromaDB data is stored unencrypted on disk
- Session memory is stored in-process
- LangSmith tracing (if enabled) sends metadata to LangSmith cloud

## Before Using Real Patient Data

Implement:

- Authentication
- Authorization
- TLS
- Encryption at rest
- Audit logging

Review all applicable HIPAA, GDPR, and local compliance requirements.
