import { useState, useRef, useEffect } from 'react'
import { sendChat, clearSession } from '../api/client'
import MessageBubble from './MessageBubble'
import './ChatWindow.css'

const QUICK_PROMPTS = [
  { label: 'Medications', text: 'What medications is the patient currently taking?' },
  { label: 'Vitals', text: 'What are the patient\'s latest vital signs?' },
  { label: 'Labs', text: 'What are the abnormal lab results?' },
  { label: 'Timeline', text: 'Walk me through the patient timeline' },
  { label: 'Diagnoses', text: 'What are the patient\'s current diagnoses?' },
]

export default function ChatWindow({ patientId, sessionId }) {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [elapsed, setElapsed] = useState(0)

  const bottomRef = useRef()
  const inputRef = useRef()
  const timerRef = useRef(null)

  // Scroll to bottom whenever messages change
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // Focus input on mount
  useEffect(() => {
    inputRef.current?.focus()
  }, [])

  // Clear messages when patient changes
  useEffect(() => {
    setMessages([])
  }, [patientId])

  // Update loading bubble elapsed time
  useEffect(() => {
    if (isLoading) {
      setMessages(prev =>
        prev.map(m =>
          m.isLoading
            ? { ...m, elapsed }
            : m
        )
      )
    }
  }, [elapsed, isLoading])

  // Cleanup timer on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current)
      }
    }
  }, [])

  async function handleSend(text) {
    const msg = (text || input).trim()
    if (!msg || isLoading) return

    setInput('')

    // Add user message immediately
    const userMsg = { role: 'user', content: msg }
    setMessages(prev => [...prev, userMsg])

    const loadingId = Date.now()

    // Start elapsed timer
    setElapsed(0)

    timerRef.current = setInterval(() => {
      setElapsed(e => e + 1)
    }, 1000)

    // Add loading placeholder
    setMessages(prev => [
      ...prev,
      {
        id: loadingId,
        role: 'assistant',
        content: '',
        isLoading: true,
        elapsed: 0,
      }
    ])

    setIsLoading(true)

    try {
      const data = await sendChat(msg, sessionId, patientId)

      // Replace loading placeholder with real response
      setMessages(prev =>
        prev.map(m =>
          m.id === loadingId
            ? {
                role: 'assistant',
                content: data.reply,
                sources: data.sources || [],
                intent: data.intent || '',
              }
            : m
        )
      )
    } catch (err) {
      const errorText =
        err.response?.data?.detail ||
        'Request failed. Is the server running?'

      setMessages(prev =>
        prev.map(m =>
          m.id === loadingId
            ? {
                role: 'assistant',
                content: `Error: ${errorText}`,
                sources: [],
                intent: '',
              }
            : m
        )
      )
    } finally {
      clearInterval(timerRef.current)
      timerRef.current = null

      setElapsed(0)
      setIsLoading(false)

      inputRef.current?.focus()
    }
  }

  async function handleClearSession() {
    try {
      await clearSession(sessionId)
    } catch (_) {
      // silently ignore — server may not have this session
    }
    setMessages([])
  }

  const isEmpty = messages.length === 0

  return (
    <div className="chat-window">
      {/* Header */}
      <div className="chat-header">
        <div className="chat-header__left">
          <h2 className="chat-title">Clinical Chat</h2>

          {patientId ? (
            <span className="chat-patient-badge">
              <span className="dot dot--green" />
              {patientId}
            </span>
          ) : (
            <span className="chat-patient-badge chat-patient-badge--none">
              No chart loaded
            </span>
          )}
        </div>

        {messages.length > 0 && (
          <button
            className="chat-clear-btn"
            onClick={handleClearSession}
            title="Clear conversation"
          >
            Clear
          </button>
        )}
      </div>

      {/* Message area */}
      <div className="chat-messages" role="log" aria-live="polite">
        {isEmpty ? (
          <div className="chat-empty">
            <p className="chat-empty__heading">
              {patientId
                ? `Chart loaded for ${patientId}`
                : 'No chart loaded'}
            </p>

            <p className="chat-empty__sub">
              {patientId
                ? 'Ask a clinical question below, or use a quick prompt'
                : 'Upload a patient chart to enable RAG mode, or ask a general question'}
            </p>

            {patientId && (
              <div className="chat-quick-prompts">
                {QUICK_PROMPTS.map((qp) => (
                  <button
                    key={qp.label}
                    className="chat-quick-btn"
                    onClick={() => handleSend(qp.text)}
                  >
                    {qp.label}
                  </button>
                ))}
              </div>
            )}
          </div>
        ) : (
          <>
            {messages.map((msg, i) => (
              <MessageBubble
                key={msg.id || i}
                message={msg}
              />
            ))}
          </>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input area */}
      <div className="chat-input-area">
        <div className="chat-input-wrap">
          <textarea
            ref={inputRef}
            className="chat-input"
            placeholder={
              patientId
                ? `Ask about ${patientId}…`
                : 'Ask a general clinical question…'
            }
            value={input}
            rows={1}
            onChange={(e) => {
              setInput(e.target.value)

              // Auto-resize
              e.target.style.height = 'auto'
              e.target.style.height =
                Math.min(e.target.scrollHeight, 120) + 'px'
            }}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                handleSend()
              }
            }}
            disabled={isLoading}
          />

          <button
            className="chat-send-btn"
            onClick={() => handleSend()}
            disabled={isLoading || !input.trim()}
            aria-label="Send message"
          >
            {isLoading ? (
              <span className="chat-send-spinner" />
            ) : (
              <span className="chat-send-arrow">↑</span>
            )}
          </button>
        </div>

        <p className="chat-input-hint">
          Enter to send · Shift+Enter for new line
          {patientId && ' · Sources shown below each response'}
        </p>
      </div>
    </div>
  )
}