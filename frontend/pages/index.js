import React, { useState } from 'react'

export default function Home() {
  const [logText, setLogText] = useState('')
  const [file, setFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  
  // NLP query states
  const [queryText, setQueryText] = useState('')
  const [nlpLoading, setNlpLoading] = useState(false)
  const [nlpResponse, setNlpResponse] = useState(null)
  const [nlpError, setNlpError] = useState(null)
  const [showNLPTab, setShowNLPTab] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const form = new FormData()
      if (file) {
        form.append('file', file)
      } else {
        form.append('log_text', logText)
      }
      form.append('model', 'transformer')

      const res = await fetch('http://localhost:8000/analyze', {
        method: 'POST',
        body: form,
      })
      const data = await res.json()
      if (!res.ok) {
        setError(data)
      } else {
        setResult(data)
      }
    } catch (err) {
      setError(err.toString())
    } finally {
      setLoading(false)
    }
  }

  const handleNLPQuery = async (e) => {
    e.preventDefault()
    if (!queryText.trim()) {
      setNlpError('Please enter a query')
      return
    }

    setNlpLoading(true)
    setNlpError(null)
    setNlpResponse(null)

    try {
      const form = new FormData()
      form.append('query', queryText)

      const res = await fetch('http://localhost:8000/query', {
        method: 'POST',
        body: form,
      })
      const data = await res.json()
      if (!res.ok || data.error) {
        setNlpError(data.error || 'Query failed')
      } else {
        setNlpResponse(data)
      }
    } catch (err) {
      setNlpError(err.toString())
    } finally {
      setNlpLoading(false)
    }
  }

  return (
    <div className="container">
      <div className="header">
        <div className="brand">
          <div className="logo">LA</div>
          <div>
            <div className="title">Cybersecurity Log Analyzer</div>
            <div className="subtitle">AI-assisted log parsing, classification, alerting & NLP queries</div>
          </div>
        </div>
        <div className="meta">Local dev UI — Backend: <code>http://localhost:8000</code></div>
      </div>

      <div style={{display:'flex',gap:12,marginBottom:12}}>
        <button 
          className={`btn ${!showNLPTab ? '' : 'secondary'}`}
          onClick={() => setShowNLPTab(false)}
          style={{flex:1}}
        >
          Log Analysis
        </button>
        <button 
          className={`btn ${showNLPTab ? '' : 'secondary'}`}
          onClick={() => setShowNLPTab(true)}
          style={{flex:1}}
        >
          NLP Queries
        </button>
      </div>

      {!showNLPTab ? (
        <div className="layout">
          <div className="card">
            <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:12}}>
              <div style={{fontWeight:800}}>Logs Input</div>
              <div style={{color:'var(--muted)',fontSize:13}}>Paste or upload a file</div>
            </div>

            <form onSubmit={handleSubmit} className="inputArea">
              <textarea
                value={logText}
                onChange={(e) => setLogText(e.target.value)}
                placeholder="Paste log lines here (one per line)"
              />

              <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginTop:12}}>
                <div className="fileInput">
                  <input type="file" accept=".log,.txt" onChange={(e) => setFile(e.target.files[0])} />
                  <div style={{color:'var(--muted)',fontSize:13}}>Optional file upload</div>
                </div>

                <div style={{display:'flex',gap:8}}>
                  <button className="btn secondary" type="button" onClick={() => { setLogText(''); setFile(null); setResult(null); setError(null)}}>Reset</button>
                  <button className="btn" type="submit" disabled={loading}>{loading ? 'Analyzing...' : 'Analyze'}</button>
                </div>
              </div>
            </form>

            {error && (
              <div style={{ marginTop: 12 }}>
                <div style={{color:'var(--danger)',fontWeight:700}}>Error</div>
                <pre style={{whiteSpace:'pre-wrap',color:'var(--muted)'}}>{JSON.stringify(error, null, 2)}</pre>
              </div>
            )}
          </div>

          <div>
            <div className="card" style={{marginBottom:16}}>
              <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:8}}>
                <div style={{fontWeight:800}}>Alerts</div>
                <div style={{color:'var(--muted)',fontSize:13}}>Detected issues</div>
              </div>

              <div className="alertsList">
                {result && result.alerts && result.alerts.length > 0 ? (
                  result.alerts.map((a, i) => (
                    <div key={i} className="alertItem">
                      <div className="alertInfo">
                        <div style={{fontWeight:700}}>[{a.alert_type}]</div>
                        <div style={{color:'var(--muted)'}}>{a.message}</div>
                      </div>
                      <div style={{display:'flex',flexDirection:'column',alignItems:'flex-end'}}>
                        <div className={`severity ${a.severity || 'Medium'}`}>{a.severity || 'Medium'}</div>
                        <div style={{fontSize:12,color:'var(--muted)',marginTop:6}}>{a.timestamp}</div>
                      </div>
                    </div>
                  ))
                ) : (
                  <div style={{color:'var(--muted)'}}>No alerts detected.</div>
                )}
              </div>
            </div>

            <div className="card">
              <div style={{fontWeight:800,marginBottom:8}}>Rows Preview</div>
              {result ? (
                <div className="preview">
                  <pre>{JSON.stringify(result.rows_preview, null, 2)}</pre>
                </div>
              ) : (
                <div style={{color:'var(--muted)'}}>No data yet. Submit logs to see parsed rows and classifications.</div>
              )}
            </div>
          </div>
        </div>
      ) : (
        <div style={{padding:20}}>
          <div className="card">
            <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:12}}>
              <div style={{fontWeight:800}}>NLP Query Interface</div>
              <div style={{color:'var(--muted)',fontSize:13}}>Ask questions about logs and alerts</div>
            </div>

            <form onSubmit={handleNLPQuery} className="inputArea">
              <textarea
                value={queryText}
                onChange={(e) => setQueryText(e.target.value)}
                placeholder="Ask questions like: 'What are the failed logins?' or 'Show suspicious activities from 192.168.0.23'"
                rows={6}
              />

              <div style={{display:'flex',justifyContent:'flex-end',gap:8,marginTop:12}}>
                <button className="btn secondary" type="button" onClick={() => setQueryText('')}>Clear</button>
                <button className="btn" type="submit" disabled={nlpLoading}>{nlpLoading ? 'Querying...' : 'Ask NLP'}</button>
              </div>
            </form>

            {nlpError && (
              <div style={{ marginTop: 12 }}>
                <div style={{color:'var(--danger)',fontWeight:700}}>Error</div>
                <pre style={{whiteSpace:'pre-wrap',color:'var(--muted)'}}>{nlpError}</pre>
              </div>
            )}

            {nlpResponse && (
              <div style={{marginTop:12}}>
                <div style={{fontWeight:800,marginBottom:8,color:'var(--accent)'}}>AI Response</div>
                <div style={{background:'var(--glass-2)',padding:12,borderRadius:8,border:'1px solid rgba(255,255,255,0.02)'}}>
                  <div style={{color:'var(--text)',lineHeight:'1.6',whiteSpace:'pre-wrap'}}>
                    {nlpResponse.response}
                  </div>
                  <div style={{marginTop:8,fontSize:12,color:'var(--muted)'}}>
                    Model: {nlpResponse.nlp_model} | Query time: {new Date(nlpResponse.timestamp).toLocaleTimeString()}
                  </div>
                </div>
              </div>
            )}
          </div>

          <footer style={{marginTop:28,color:'var(--muted)',fontSize:13,textAlign:'center'}}>
            <div style={{maxWidth:900,margin:'8px auto'}}>Backend: <code>uvicorn src.api:app --reload --port 8000</code> | Frontend: <code>npm run dev</code></div>
          </footer>
        </div>
      )}

      {!showNLPTab && (
        <footer style={{marginTop:28,color:'var(--muted)',fontSize:13,textAlign:'center'}}>
          <div style={{maxWidth:900,margin:'8px auto'}}>Backend: <code>uvicorn src.api:app --reload --port 8000</code> | Frontend: <code>npm run dev</code></div>
        </footer>
      )}
    </div>
  )
}
