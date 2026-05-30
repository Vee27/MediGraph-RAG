import { useState, useEffect, useCallback } from 'react'
import { fetchHealth } from './api/client'
import UploadPanel from './components/UploadPanel'
import ChatWindow from './components/ChatWindow'
import SOAPViewer from './components/SOAPViewer'
import './App.css'

// Generate a stable session ID for this browser tab
function makeSessionId() {
  return 'session-' + Math.random().toString(36).slice(2, 9)
}

const SESSION_ID = makeSessionId()

// Three tabs in the right panel
const RIGHT_TABS = [
  { id: 'chat', label: 'Chat'      },
  { id: 'soap', label: 'SOAP Note' },
]

export default function App() {
  const [activePatientId, setActivePatientId] = useState('')
  const [rightTab, setRightTab]               = useState('chat')
  const [health, setHealth]                   = useState(null)  // null | { status, ollama_reachable, model }
  const [sidebarOpen, setSidebarOpen]         = useState(true)

  // Poll health on mount
  useEffect(() => {
    async function check() {
      try {
        const data = await fetchHealth()
        setHealth(data)
      } catch {
        setHealth({ status: 'unreachable', ollama_reachable: false, model: '—' })
      }
    }
    check()
    const id = setInterval(check, 30_000)
    return () => clearInterval(id)
  }, [])

  const handleUploadSuccess = useCallback((patientId) => {
    setActivePatientId(patientId)
    setRightTab('chat')
  }, [])

  const isHealthy = health?.status === 'ok' && health?.ollama_reachable

  return (
    <div className="app">

      {/* ── Top bar ─────────────────────────────────────────── */}
      <header className="topbar">
        <div className="topbar__left">
          <button
            className="topbar__sidebar-toggle"
            onClick={() => setSidebarOpen(o => !o)}
            aria-label="Toggle sidebar"
          >
            ☰
          </button>
          <div className="topbar__brand">
            <span className="topbar__brand-m">Medi</span>
            <span className="topbar__brand-g">Graph</span>
            <span className="topbar__brand-tag">RAG</span>
          </div>
        </div>

        <div className="topbar__right">
          {/* Session */}
          <div className="topbar__meta">
            <span className="topbar__meta-label">session</span>
            <span className="topbar__meta-value">{SESSION_ID}</span>
          </div>

          {/* Model */}
          {health && (
            <div className="topbar__meta">
              <span className="topbar__meta-label">model</span>
              <span className="topbar__meta-value">{health.model}</span>
            </div>
          )}

          {/* Health indicator */}
          <div className={`topbar__health ${isHealthy ? 'topbar__health--ok' : 'topbar__health--err'}`}>
            <span className={`dot ${isHealthy ? 'dot--green' : 'dot--red'}`} />
            {health === null ? 'Checking…' : isHealthy ? 'Online' : 'Offline'}
          </div>
        </div>
      </header>

      {/* ── Main layout ─────────────────────────────────────── */}
      <div className="app__body">

        {/* Left sidebar — upload */}
        <aside className={`sidebar ${sidebarOpen ? 'sidebar--open' : 'sidebar--closed'}`}>
          <UploadPanel
            onUploadSuccess={handleUploadSuccess}
            activePatientId={activePatientId}
          />
        </aside>

        {/* Right panel — chat / SOAP */}
        <main className="main-panel">

          {/* Tab bar */}
          <div className="tab-bar">
            {RIGHT_TABS.map((tab) => (
              <button
                key={tab.id}
                className={`tab-btn ${rightTab === tab.id ? 'tab-btn--active' : ''}`}
                onClick={() => setRightTab(tab.id)}
              >
                {tab.label}
                {tab.id === 'chat' && activePatientId && (
                  <span className="tab-btn__dot" />
                )}
              </button>
            ))}

            {/* Active patient shown in tab bar */}
            {activePatientId && (
              <div className="tab-bar__patient">
                <span className="dot dot--green" style={{ width: 6, height: 6 }} />
                {activePatientId}
              </div>
            )}
          </div>

          {/* Panel content */}
          <div className="panel-content">
            {rightTab === 'chat' && (
              <ChatWindow
                patientId={activePatientId}
                sessionId={SESSION_ID}
              />
            )}
            {rightTab === 'soap' && (
              <SOAPViewer patientId={activePatientId} />
            )}
          </div>
        </main>
      </div>
    </div>
  )
}
