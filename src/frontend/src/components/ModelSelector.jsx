import './ModelSelector.css'

export default function ModelSelector({ models, selected, onChange }) {
  const getModelType = (name) => {
    const n = name.toLowerCase()
    if (n.includes('seg'))    return { label: 'Seg',   color: '#a78bfa' }
    if (n.includes('multi'))  return { label: 'Multi', color: '#34d399' }
    return                           { label: 'Single',color: '#60a5fa' }
  }

  return (
    <div className="model-selector">
      <label className="model-label">Model</label>
      {models.length === 0 ? (
        <div className="model-empty">No models found — place .pt files in the project root</div>
      ) : (
        <div className="select-wrapper">
          <select
            value={selected}
            onChange={e => onChange(e.target.value)}
            className="model-select"
          >
            {models.map(m => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
          {selected && (
            <span
              className="model-type-badge"
              style={{ background: getModelType(selected).color + '22',
                       color: getModelType(selected).color,
                       borderColor: getModelType(selected).color + '55' }}
            >
              {getModelType(selected).label}
            </span>
          )}
        </div>
      )}
    </div>
  )
}
