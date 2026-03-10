import './ResultsPanel.css'

const CONF_COLORS = { high: '#22c55e', mid: '#f59e0b', low: '#ef4444' }

function DiagnosisBadge({ diagnosis, probability }) {
  const isMalignant = probability >= 0.5
  const conf = isMalignant ? probability : 1 - probability
  const confColor = conf >= .80 ? CONF_COLORS.high : conf >= .60 ? CONF_COLORS.mid : CONF_COLORS.low
  const pct = Math.round(conf * 100)

  return (
    <div className={`diagnosis-card ${isMalignant ? 'malignant' : 'benign'}`}>
      <div className="diag-left">
        <span className="diag-dot" />
        <div>
          <p className="diag-label">{isMalignant ? 'MALIGNANT' : 'BENIGN'}</p>
          <p className="diag-text">{diagnosis}</p>
        </div>
      </div>
      <div className="conf-ring" style={{ '--pct': pct, '--color': confColor }}>
        <svg viewBox="0 0 36 36">
          <circle cx="18" cy="18" r="15.9" fill="none" stroke="rgba(255,255,255,.08)" strokeWidth="3" />
          <circle
            cx="18" cy="18" r="15.9"
            fill="none"
            stroke={confColor}
            strokeWidth="3"
            strokeDasharray={`${pct} 100`}
            strokeLinecap="round"
            transform="rotate(-90 18 18)"
          />
        </svg>
        <span className="conf-value" style={{ color: confColor }}>{pct}%</span>
      </div>
    </div>
  )
}

function ImageCard({ title, src, tag }) {
  if (!src) return null
  return (
    <div className="image-card">
      <div className="image-card-header">
        <span className="image-card-title">{title}</span>
        {tag && <span className="image-tag">{tag}</span>}
      </div>
      <div className="image-wrapper">
        <img src={`data:image/png;base64,${src}`} alt={title} />
      </div>
    </div>
  )
}

export default function ResultsPanel({ results }) {
  const { model_type } = results

  return (
    <div className="results-panel">
      <h2 className="results-title">Analysis Results</h2>
      <DiagnosisBadge diagnosis={results.diagnosis} probability={results.probability} />

      {model_type === 'single' && (
        <div className="image-grid two-col">
          <ImageCard title="Original Mammogram" src={results.original_cc} tag="Input" />
          <ImageCard title="GradCAM Heatmap" src={results.heatmap_cc} tag="XAI" />
        </div>
      )}

      {model_type === 'multi' && (
        <>
          <div className="view-section">
            <h3 className="view-label">CC View</h3>
            <div className="image-grid two-col">
              <ImageCard title="Original" src={results.original_cc} tag="Input" />
              <ImageCard title="GradCAM Heatmap" src={results.heatmap_cc} tag="XAI" />
            </div>
          </div>
          <div className="view-section">
            <h3 className="view-label">MLO View</h3>
            <div className="image-grid two-col">
              <ImageCard title="Original" src={results.original_mlo} tag="Input" />
              <ImageCard title="GradCAM Heatmap" src={results.heatmap_mlo} tag="XAI" />
            </div>
          </div>
        </>
      )}

      {model_type === 'multi_seg' && (
        <>
          {!results.seg_mask_cc && (
            <div className="benign-note">
              No lesion localisation — model predicts benign, ROI overlay not applicable.
            </div>
          )}
          <div className="view-section">
            <h3 className="view-label">CC View</h3>
            <div className={`image-grid ${results.seg_mask_cc ? 'two-col' : 'one-col'}`}>
              <ImageCard title="Original" src={results.original_cc} tag="Input" />
              {results.seg_mask_cc && <ImageCard title="ROI Prediction" src={results.seg_mask_cc} tag="Seg Head" />}
            </div>
          </div>
          <div className="view-section">
            <h3 className="view-label">MLO View</h3>
            <div className={`image-grid ${results.seg_mask_mlo ? 'two-col' : 'one-col'}`}>
              <ImageCard title="Original" src={results.original_mlo} tag="Input" />
              {results.seg_mask_mlo && <ImageCard title="ROI Prediction" src={results.seg_mask_mlo} tag="Seg Head" />}
            </div>
          </div>
        </>
      )}
    </div>
  )
}
