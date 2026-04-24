import { useState, useEffect } from 'react'
import axios from 'axios'
import ModelSelector from './components/ModelSelector'
import UploadArea from './components/UploadArea'
import ResultsPanel from './components/ResultsPanel'
import './App.css'

export default function App() {
  const [models, setModels]           = useState([])
  const [selectedModel, setSelectedModel] = useState('')
  const [ccFile, setCcFile]           = useState(null)
  const [mloFile, setMloFile]         = useState(null)
  const [loading, setLoading]         = useState(false)
  const [results, setResults]         = useState(null)
  const [error, setError]             = useState(null)

  const isMultiView = selectedModel.toLowerCase().startsWith('mv')
  const canSubmit   = selectedModel && ccFile && (!isMultiView || mloFile)

  useEffect(() => {
    axios.get('/api/models')
      .then(r => {
        setModels(r.data.models)
        if (r.data.models.length > 0) setSelectedModel(r.data.models[0])
      })
      .catch(() => setError('Could not connect to backend. Is the API server running?'))
  }, [])

  const handleModelChange = (m) => {
    setSelectedModel(m)
    setResults(null)
    setError(null)
    setCcFile(null)
    setMloFile(null)
  }

  const handleAnalyse = async () => {
    setLoading(true)
    setResults(null)
    setError(null)

    const form = new FormData()
    form.append('model_name', selectedModel)
    form.append('cc_file', ccFile)
    if (isMultiView && mloFile) form.append('mlo_file', mloFile)

    try {
      const { data } = await axios.post('/api/predict', form, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      setResults(data)
    } catch (e) {
      setError(e.response?.data?.detail || 'Inference failed. Check the console for details.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-content">
          <div className="logo">
            <span className="logo-icon">⬡</span>
            <div>
              <h1>MammogramXAI</h1>
              <p className="subtitle">CBIS-DDSM Multi-View Explainability</p>
            </div>
          </div>
          <div className="header-badge">ResNet18 · CBIS-DDSM</div>
        </div>
      </header>

      <main className="app-main">
        <div className="config-row">
          <ModelSelector
            models={models}
            selected={selectedModel}
            onChange={handleModelChange}
          />
          <button
            className={`btn-analyse ${loading ? 'loading' : ''}`}
            onClick={handleAnalyse}
            disabled={!canSubmit || loading}
          >
            {loading ? (
              <><span className="spinner" /> Analysing…</>
            ) : (
              <><span>▶</span> Analyse</>
            )}
          </button>
        </div>

        <div className={`upload-row ${isMultiView ? 'two-col' : 'one-col'}`}>
          <UploadArea
            label={isMultiView ? 'CC View' : 'Mammogram'}
            hint="Craniocaudal"
            file={ccFile}
            onFileSelect={f => { setCcFile(f); setResults(null) }}
          />
          {isMultiView && (
            <UploadArea
              label="MLO View"
              hint="Mediolateral Oblique"
              file={mloFile}
              onFileSelect={f => { setMloFile(f); setResults(null) }}
            />
          )}
        </div>

        {error && (
          <div className="error-banner">
            <span>⚠</span> {error}
          </div>
        )}

        {results && <ResultsPanel results={results} />}
      </main>
    </div>
  )
}
