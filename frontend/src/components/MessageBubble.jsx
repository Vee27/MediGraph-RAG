import ReactMarkdown from 'react-markdown'
import './MessageBubble.css'

const INTENT_LABELS = {
  retrieve:   { label: 'RAG',        color: 'blue'  },
  medication: { label: 'Medication', color: 'green' },
  timeline:   { label: 'Timeline',   color: 'amber' },
  soap:       { label: 'SOAP',       color: 'purple'},
  direct:     { label: 'Direct',     color: 'muted' },
}

export default function MessageBubble({ message }) {
  const { role, content, sources, intent, isLoading } = message
  const isUser = role === 'user'
  const intentMeta = intent ? INTENT_LABELS[intent] : null

  return (
    <div className={`bubble-wrap ${isUser ? 'bubble-wrap--user' : 'bubble-wrap--assistant'}`}>
      {/* Avatar */}
      <div className={`bubble-avatar ${isUser ? 'bubble-avatar--user' : 'bubble-avatar--assistant'}`}
        aria-hidden="true"
      >
        {isUser ? 'MD' : 'MG'}
      </div>

      <div className="bubble-body">
        {/* Intent tag — only on assistant messages */}
        {!isUser && intentMeta && (
          <span className={`bubble-intent bubble-intent--${intentMeta.color}`}>
            {intentMeta.label}
          </span>
        )}

        {/* Message content */}
        <div className={`bubble-content ${isUser ? 'bubble-content--user' : 'bubble-content--assistant'}`}>
          {isLoading ? (
            <div className="bubble-loading" aria-label="Thinking">
            <span />
            <span />
            <span />

            <div className="bubble-loading-text">
              {message.elapsed < 15
                ? `Searching chart... ${message.elapsed}s`
                : message.elapsed < 45
                ? `Generating response... ${message.elapsed}s`
                : `Still thinking... ${message.elapsed}s`}
            </div>
          </div>
          ) : isUser ? (
            <p>{content}</p>
          ) : (
            <ReactMarkdown>{content}</ReactMarkdown>
          )}
        </div>

        {/* Sources */}
        {!isUser && sources && sources.length > 0 && (
          <details className="bubble-sources">
            <summary className="bubble-sources__toggle">
              {sources.length} source{sources.length > 1 ? 's' : ''}
            </summary>
            <div className="bubble-sources__list">
              {sources.map((s, i) => (
                <div key={i} className="bubble-source-item">
                  <div className="bubble-source-item__meta">
                    <span className="bubble-source-item__page">Page {s.page}</span>
                    <span className="bubble-source-item__score">
                      {(s.score * 100).toFixed(0)}% match
                    </span>
                  </div>
                  <p className="bubble-source-item__text">{s.text}</p>
                </div>
              ))}
            </div>
          </details>
        )}
      </div>
    </div>
  )
}
