import { useState } from 'react'
import { generateSOAP } from '../api/client'
import ReactMarkdown from 'react-markdown'
import './SOAPViewer.css'

const SECTIONS = ['SUBJECTIVE', 'OBJECTIVE', 'ASSESSMENT', 'PLAN']

function parseSOAPSections(text) {
  const result = {}
  let current = null
  const lines = text.split('\n')

  for (const line of lines) {
    const trimmed = line.trim()
    const matched = SECTIONS.find(s =>
      trimmed.toUpperCase().startsWith(s + ':') ||
      trimmed.toUpperCase() === s
    )
    if (matched) {
      current = matched
      // capture inline content after "SUBJECTIVE:"
      const inline = trimmed.slice(matched.length).replace(/^:\s*/, '').trim()
      result[matched] = inline ? [inline] : []
    } else if (current && trimmed) {
      result[current] = result[current] || []
      result[current].push(trimmed)
    }
  }
  return result
}

const SECTION_META = {
  SUBJECTIVE:  { label: 'S', desc: 'Subjective',  color: 'blue'   },
  OBJECTIVE:   { label: 'O', desc: 'Objective',   color: 'green'  },
  ASSESSMENT:  { label: 'A', desc: 'Assessment',  color: 'amber'  },
  PLAN:        { label: 'P', desc: 'Plan',         color: 'purple' },
}

export default function SOAPViewer({ patientId }) {
  const [status, setStatus]   = useState('idle')   // idle | loading | done | error
  const [rawNote, setRawNote] = useState('')
  const [parsed, setParsed]   = useState(null)
  const [error, setError]     = useState('')
  const [view, setView]       = useState('structured')  // structured | raw

  async function handleGenerate() {
    if (!patientId) return
    setStatus('loading')
    setError('')

    try {
      const data = await generateSOAP(patientId)
      setRawNote(data.soap_note)
      setParsed(parseSOAPSections(data.soap_note))
      setStatus('done')
    } catch (err) {
      setError(err.response?.data?.detail || 'SOAP generation failed.')
      setStatus('error')
    }
  }

  function handleCopy() {
    navigator.clipboard.writeText(rawNote)
  }

  if (!patientId) {
    return (
      <div className="soap-viewer soap-viewer--empty">
        <p className="soap-empty-msg">Upload a chart to generate a SOAP note</p>
      </div>
    )
  }

  return (
    <div className="soap-viewer">
      <div className="soap-header">
        <div className="soap-header__left">
          <h2 className="soap-title">SOAP Note</h2>
          <span className="soap-patient-id">{patientId}</span>
        </div>

        <div className="soap-header__right">
          {status === 'done' && (
            <>
              <div className="soap-view-toggle">
                <button
                  className={`soap-toggle-btn ${view === 'structured' ? 'active' : ''}`}
                  onClick={() => setView('structured')}
                >Structured</button>
                <button
                  className={`soap-toggle-btn ${view === 'raw' ? 'active' : ''}`}
                  onClick={() => setView('raw')}
                >Raw</button>
              </div>
              <button className="soap-copy-btn" onClick={handleCopy}>
                Copy
              </button>
            </>
          )}
          <button
            className="soap-generate-btn"
            onClick={handleGenerate}
            disabled={status === 'loading'}
          >
            {status === 'loading' ? 'Generating…' : status === 'done' ? 'Regenerate' : 'Generate'}
          </button>
        </div>
      </div>

      {/* Error */}
      {status === 'error' && (
        <p className="soap-error">{error}</p>
      )}

      {/* Loading */}
      {status === 'loading' && (
        <div className="soap-loading">
          <div className="soap-loading__bar" />
          <p>Analysing chart and generating SOAP note…</p>
        </div>
      )}

      {/* Content */}
      {status === 'done' && view === 'structured' && parsed && (
        <div className="soap-sections">
          {SECTIONS.map((key) => {
            const meta  = SECTION_META[key]
            const lines = parsed[key] || []
            return (
              <div key={key} className={`soap-section soap-section--${meta.color}`}>
                <div className="soap-section__header">
                  <span className={`soap-section__badge soap-section__badge--${meta.color}`}>
                    {meta.label}
                  </span>
                  <span className="soap-section__title">{meta.desc}</span>
                </div>
                <div className="soap-section__body">
                  {lines.length > 0 ? (
                    <ul className="soap-section__list">
                      {lines.map((line, i) => (
                        <li key={i} className="soap-section__item">
                          {line.replace(/^[•\-*]\s*/, '')}
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="soap-section__empty">Not documented</p>
                  )}
                </div>
              </div>
            )
          })}
        </div>
      )}

      {status === 'done' && view === 'raw' && (
        <div className="soap-raw">
          <ReactMarkdown>{rawNote}</ReactMarkdown>
        </div>
      )}

      {/* Idle state */}
      {status === 'idle' && (
        <div className="soap-idle">
          <div className="soap-idle__grid">
            {SECTIONS.map((s) => (
              <div key={s} className="soap-idle__cell">
                <span className={`soap-idle__letter soap-idle__letter--${SECTION_META[s].color}`}>
                  {SECTION_META[s].label}
                </span>
                <span className="soap-idle__desc">{SECTION_META[s].desc}</span>
              </div>
            ))}
          </div>
          <p className="soap-idle__hint">Click Generate to create a SOAP note from the uploaded chart</p>
        </div>
      )}
    </div>
  )
}
