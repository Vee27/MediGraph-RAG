import { useState, useRef } from 'react'
import { uploadChart } from '../api/client'
import './UploadPanel.css'

export default function UploadPanel({ onUploadSuccess, activePatientId }) {
  const [file, setFile]           = useState(null)
  const [patientId, setPatientId] = useState('')
  const [status, setStatus]       = useState('idle')  // idle | uploading | success | error
  const [progress, setProgress]   = useState(0)
  const [result, setResult]       = useState(null)
  const [error, setError]         = useState('')
  const [dragging, setDragging]   = useState(false)
  const inputRef = useRef()

  function handleFile(f) {
    if (!f) return
    if (f.type !== 'application/pdf') {
      setError('Only PDF files are supported.')
      return
    }
    setFile(f)
    setError('')
    setStatus('idle')
    setResult(null)
  }

  function onDrop(e) {
    e.preventDefault()
    setDragging(false)
    handleFile(e.dataTransfer.files[0])
  }

  async function handleUpload() {
    if (!file)      return setError('Please select a PDF file.')
    if (!patientId.trim()) return setError('Please enter a patient ID.')

    setStatus('uploading')
    setProgress(0)
    setError('')

    try {
      const data = await uploadChart(file, patientId.trim(), setProgress)
      setResult(data)
      setStatus('success')
      onUploadSuccess?.(patientId.trim(), data)
    } catch (err) {
      setError(err.response?.data?.detail || 'Upload failed. Is the server running?')
      setStatus('error')
    }
  }

  function reset() {
    setFile(null)
    setPatientId('')
    setStatus('idle')
    setProgress(0)
    setResult(null)
    setError('')
  }

  return (
    <div className="upload-panel">
      <div className="upload-panel__header">
        <span className="upload-panel__icon">⊕</span>
        <h2 className="upload-panel__title">Upload Chart</h2>
      </div>

      {/* Active patient badge */}
      {activePatientId && (
        <div className="upload-panel__active-badge">
          <span className="dot dot--green" />
          Active: <strong>{activePatientId}</strong>
        </div>
      )}

      {/* Drop zone */}
      <div
        className={`upload-dropzone ${dragging ? 'upload-dropzone--drag' : ''} ${file ? 'upload-dropzone--has-file' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => e.key === 'Enter' && inputRef.current?.click()}
        aria-label="Drop PDF here or click to browse"
      >
        <input
          ref={inputRef}
          type="file"
          accept=".pdf"
          className="sr-only"
          onChange={(e) => handleFile(e.target.files[0])}
        />
        {file ? (
          <div className="upload-dropzone__file">
            <span className="upload-dropzone__file-icon">📄</span>
            <span className="upload-dropzone__file-name">{file.name}</span>
            <span className="upload-dropzone__file-size">
              {(file.size / 1024).toFixed(1)} KB
            </span>
          </div>
        ) : (
          <div className="upload-dropzone__prompt">
            <span className="upload-dropzone__arrow">↑</span>
            <p>Drop PDF here</p>
            <p className="upload-dropzone__sub">or click to browse</p>
          </div>
        )}
      </div>

      {/* Patient ID */}
      <div className="upload-field">
        <label className="upload-label" htmlFor="patient-id">
          Patient ID
        </label>
        <input
          id="patient-id"
          className="upload-input"
          type="text"
          placeholder="e.g. patient-001"
          value={patientId}
          onChange={(e) => setPatientId(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleUpload()}
        />
      </div>

      {/* Progress bar */}
      {status === 'uploading' && (
        <div className="upload-progress">
          <div className="upload-progress__track">
            <div
              className="upload-progress__fill"
              style={{ width: `${progress}%` }}
            />
          </div>
          <span className="upload-progress__label">
            {progress < 100 ? `Uploading… ${progress}%` : 'Processing…'}
          </span>
        </div>
      )}

      {/* Error */}
      {error && (
        <p className="upload-error">{error}</p>
      )}

      {/* Success result */}
      {status === 'success' && result && (
        <div className="upload-result">
          <div className="upload-result__row">
            <span className="upload-result__label">Pages</span>
            <span className="upload-result__value">{result.pages_processed}</span>
          </div>
          <div className="upload-result__row">
            <span className="upload-result__label">Chunks stored</span>
            <span className="upload-result__value">{result.chunks_stored}</span>
          </div>
          <div className="upload-result__row">
            <span className="upload-result__label">Patient ID</span>
            <span className="upload-result__value">{result.patient_id}</span>
          </div>
        </div>
      )}

      {/* Actions */}
      <div className="upload-actions">
        {status === 'success' ? (
          <button className="btn btn--ghost" onClick={reset}>
            Upload another
          </button>
        ) : (
          <button
            className="btn btn--primary"
            onClick={handleUpload}
            disabled={status === 'uploading' || !file || !patientId}
          >
            {status === 'uploading' ? 'Uploading…' : 'Index chart'}
          </button>
        )}
      </div>
    </div>
  )
}
