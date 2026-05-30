import axios from 'axios'

const http = axios.create({
  baseURL: '/api',
  timeout: 120_000,   // 2 min — Ollama can be slow on first call
  headers: { 'Content-Type': 'application/json' },
})

// ── Health ────────────────────────────────────────────────────
export async function fetchHealth() {
  const { data } = await http.get('/health')
  return data
}

// ── Upload ────────────────────────────────────────────────────
/**
 * Upload a PDF chart for a patient.
 * @param {File}   file
 * @param {string} patientId
 * @param {function} onProgress  — callback(percent 0-100)
 */
export async function uploadChart(file, patientId, onProgress) {
  const form = new FormData()
  form.append('file', file)
  form.append('patient_id', patientId)

  const { data } = await http.post('/upload', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: (e) => {
      if (onProgress && e.total) {
        onProgress(Math.round((e.loaded / e.total) * 100))
      }
    },
  })
  return data
  // returns { message, patient_id, chunks_stored, pages_processed }
}

// ── Chat ──────────────────────────────────────────────────────
/**
 * Send a chat message.
 * @param {string} message
 * @param {string} sessionId
 * @param {string} patientId  — empty string = direct Ollama mode
 */
export async function sendChat(message, sessionId, patientId = '') {
  const { data } = await http.post('/chat', {
    message,
    session_id: sessionId,
    patient_id: patientId,
  })
  return data
  // returns { reply, session_id, model, sources, intent }
}

// ── SOAP ──────────────────────────────────────────────────────
/**
 * Generate a SOAP note for an uploaded patient chart.
 * @param {string} patientId
 * @param {string} sessionId
 */
export async function generateSOAP(patientId, sessionId = 'default') {
  const { data } = await http.post('/soap', {
    patient_id: patientId,
    session_id: sessionId,
  })
  return data
  // returns { soap_note, patient_id }
}

// ── Session ───────────────────────────────────────────────────
export async function clearSession(sessionId) {
  const { data } = await http.delete(`/session/${sessionId}`)
  return data
}

export async function fetchSessionStats(sessionId) {
  const { data } = await http.get(`/session/${sessionId}/stats`)
  return data
}
