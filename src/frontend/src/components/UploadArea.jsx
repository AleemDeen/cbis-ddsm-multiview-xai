import { useRef, useState } from 'react'
import './UploadArea.css'

export default function UploadArea({ label, hint, file, onFileSelect }) {
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef()

  const handleDrop = (e) => {
    e.preventDefault()
    setDragging(false)
    const f = e.dataTransfer.files[0]
    if (f) onFileSelect(f)
  }

  return (
    <div
      className={`upload-area ${dragging ? 'dragging' : ''} ${file ? 'has-file' : ''}`}
      onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      onClick={() => inputRef.current.click()}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => e.key === 'Enter' && inputRef.current.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".dcm,application/dicom"
        onChange={(e) => e.target.files[0] && onFileSelect(e.target.files[0])}
      />

      {file ? (
        <div className="file-selected">
          <span className="file-icon">🗂</span>
          <div className="file-info">
            <p className="file-name">{file.name}</p>
            <p className="file-size">{(file.size / 1024).toFixed(1)} KB · Click to replace</p>
          </div>
          <span className="check">✓</span>
        </div>
      ) : (
        <div className="upload-prompt">
          <div className="upload-icon">⬆</div>
          <div className="upload-text">
            <p className="upload-title">{label}</p>
            {hint && <p className="upload-hint">{hint}</p>}
            <p className="upload-sub">Drag &amp; drop or <span className="link">browse</span></p>
            <p className="upload-fmt">.dcm DICOM files only</p>
          </div>
        </div>
      )}
    </div>
  )
}
