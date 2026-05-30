# API Examples

## Upload a chart

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@datasets/sample_charts/sample_chart.pdf" \
  -F "patient_id=patient-001"
```

## Chat

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message":"What medications is the patient on?",
    "session_id":"s1",
    "patient_id":"patient-001"
  }'
```

## Generate SOAP Note

```bash
curl -X POST http://localhost:8000/soap \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id":"patient-001"
  }'
```

## Clear Session

```bash
curl -X DELETE http://localhost:8000/session/s1
```

Interactive Swagger documentation:

http://localhost:8000/docs
