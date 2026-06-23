import { useState, useRef, useCallback } from "react";

const API_BASE = "http://127.0.0.1:8080";

const palette = {
  bg: "#F0F4F8",
  surface: "#FFFFFF",
  navy: "#0B2545",
  navyMid: "#1B4B82",
  teal: "#0EA5C9",
  present: "#DC2626",
  absent: "#16A34A",
  conflict: "#D97706",
  conflictBg: "#FEF3C7",
  defer: "#DC2626",
  proceed: "#16A34A",
  labLow: "#2563EB",
  labNormal: "#16A34A",
  labHigh: "#DC2626",
  border: "#CBD5E1",
  muted: "#64748B",
  labelText: "#374151",
};

function predictionColor(val) {
  if (!val) return [palette.border, palette.muted];
  return val === "PRESENT" ? ["#FEE2E2", palette.present] : ["#DCFCE7", palette.absent];
}

function labColor(status) {
  if (status === "high") return palette.labHigh;
  if (status === "low") return palette.labLow;
  return palette.labNormal;
}

function labBg(status) {
  if (status === "high") return "#FEE2E2";
  if (status === "low") return "#DBEAFE";
  return "#DCFCE7";
}

function imageTypeColor(type) {
  if (type === "COVID") return ["#FEE2E2", "#DC2626"];
  if (type === "Viral Pneumonia") return ["#FEF3C7", "#D97706"];
  if (type === "Lung Opacity") return ["#F3E8FF", "#7C3AED"];
  return ["#DCFCE7", "#16A34A"];
}

function Spinner() {
  return <div style={{
    width: 18, height: 18,
    border: "2.5px solid rgba(255,255,255,0.3)",
    borderTop: "2.5px solid #fff",
    borderRadius: "50%",
    animation: "spin 0.7s linear infinite",
  }} />;
}

function SectionTitle({ children }) {
  return (
    <p style={{
      fontSize: 13, fontWeight: 700, color: palette.navyMid,
      textTransform: "uppercase", letterSpacing: "0.8px",
      marginBottom: 16, paddingBottom: 10,
      borderBottom: `1px solid ${palette.border}`,
      marginTop: 0,
    }}>{children}</p>
  );
}

function ResultRow({ label, children, noBorder }) {
  return (
    <div style={{
      display: "flex", alignItems: "center",
      justifyContent: "space-between",
      padding: "12px 0",
      borderBottom: noBorder ? "none" : `1px solid ${palette.border}`,
    }}>
      <span style={{ fontSize: 13, color: palette.muted, fontWeight: 500 }}>{label}</span>
      {children}
    </div>
  );
}

function Badge({ bg, color, children }) {
  return (
    <span style={{
      background: bg, color, padding: "4px 12px",
      borderRadius: 20, fontSize: 13, fontWeight: 700, letterSpacing: "0.4px",
    }}>{children}</span>
  );
}

function ProgressBar({ value, color }) {
  return (
    <div style={{ flex: 1, marginLeft: 10, background: "#F1F5F9", borderRadius: 4, height: 8, overflow: "hidden" }}>
      <div style={{
        width: `${(value * 100).toFixed(1)}%`,
        height: "100%", background: color,
        borderRadius: 4, transition: "width 0.4s ease",
      }} />
    </div>
  );
}

export default function App() {
  const [image, setImage] = useState(null);
  const [imageFile, setImageFile] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [labValue, setLabValue] = useState("7.5");
  const [refLow, setRefLow] = useState("4.5");
  const [refHigh, setRefHigh] = useState("11.0");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [apiOk, setApiOk] = useState(null);
  const fileRef = useRef();

  useState(() => {
    fetch(`${API_BASE}/health`)
      .then((r) => r.json())
      .then((d) => setApiOk(d.status === "ok"))
      .catch(() => setApiOk(false));
  }, []);

  const handleFile = useCallback((file) => {
    if (!file || !file.type.startsWith("image/")) return;
    setImageFile(file);
    setImage(URL.createObjectURL(file));
    setResult(null);
    setError(null);
  }, []);

  const onDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    handleFile(e.dataTransfer.files[0]);
  };

  const onSubmit = async () => {
    if (!imageFile) return;
    setLoading(true);
    setError(null);
    setResult(null);
    const form = new FormData();
    form.append("image", imageFile);
    form.append("lab_value", parseFloat(labValue));
    form.append("ref_low", parseFloat(refLow));
    form.append("ref_high", parseFloat(refHigh));
    try {
      const res = await fetch(`${API_BASE}/predict`, { method: "POST", body: form });
      if (!res.ok){
        if (res.status === 400) {
          const errData = await res.json();
          throw new Error(errData.detail.message || "Invalid input data.");
        }
        throw new Error(`Server returned ${res.status}`);
      }
      setResult(await res.json());
    } catch (err) {
      setError(err.message || "Request failed. Is the API running?");
    } finally {
      setLoading(false);
    }
  };

  const conflictDetected = result?.conflict_detected;
  const deferGate = result?.defer_gate === "DEFER";
  const pathologyScores = result?.model?.pathology_scores
    ? Object.entries(result.model.pathology_scores).sort((a, b) => b[1] - a[1])
    : [];
  const confidenceScores = result?.image_type_confidence
    ? Object.entries(result.image_type_confidence).sort((a, b) => b[1] - a[1])
    : [];

  const cardStyle = {
    background: palette.surface, borderRadius: 12,
    boxShadow: "0 1px 3px rgba(0,0,0,0.08), 0 4px 16px rgba(0,0,0,0.06)",
    padding: 24,
  };

  const inputStyle = {
    width: "100%", padding: "10px 14px",
    border: `1.5px solid ${palette.border}`, borderRadius: 8,
    fontSize: 14, color: palette.navy, outline: "none",
    boxSizing: "border-box", background: "#FAFBFC",
  };


  const inputStyleMuted = {
    width: "100%", padding: "10px 14px",
    border: `1.5px solid ${palette.border}`, borderRadius: 8,
    fontSize: 14, color: palette.muted, outline: "none",
    boxSizing: "border-box", background: "#FAFBFC",
  };


  const labelStyle = {
    display: "block", fontSize: 12, fontWeight: 600,
    color: palette.labelText, marginBottom: 6, letterSpacing: "0.3px",
  };

  return (
    <div style={{ minHeight: "100vh", background: palette.bg, fontFamily: "'Inter','Segoe UI',sans-serif", color: palette.navy }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        input:focus { border-color: #1B4B82 !important; background: #fff !important; outline: none; }
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
        .fade { animation: fadeIn 0.3s ease; }
      `}</style>

      {/* Header */}
      <div style={{ background: palette.navy, padding: "18px 40px", display: "flex", alignItems: "center", gap: 14, boxShadow: "0 2px 8px rgba(0,0,0,0.2)" }}>
        <div style={{ width: 38, height: 38, background: palette.teal, borderRadius: 8, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 20 }}>🫁</div>
        <div>
          <p style={{ fontSize: 18, fontWeight: 700, color: "#fff", letterSpacing: "-0.3px" }}>Clinical VLM Conflict Detection</p>
          <p style={{ fontSize: 12, color: "#94A3B8", marginTop: 2 }}>
            <span style={{ width: 8, height: 8, borderRadius: "50%", background: apiOk ? "#22C55E" : "#EF4444", display: "inline-block", marginRight: 6 }} />
            {apiOk === null ? "Checking API..." : apiOk ? "API connected" : "API unreachable — start your server"}
          </p>
        </div>
      </div>

      {/* Main grid */}
      <div style={{ maxWidth: 1200, margin: "0 auto", padding: "28px 24px", display: "grid", gridTemplateColumns: "380px 1fr", gap: 24, alignItems: "start" }}>

        {/* ── Left: Input ── */}
        <div style={cardStyle}>
          <SectionTitle>Input</SectionTitle>

          {/* Upload zone */}
          <div
            onClick={() => fileRef.current.click()}
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            style={{
              border: `2px dashed ${dragging ? palette.teal : image ? palette.navyMid : palette.border}`,
              borderRadius: 10, padding: "24px 16px", textAlign: "center", cursor: "pointer",
              background: dragging ? "#EFF6FF" : image ? "#F0F9FF" : "#FAFBFC",
              transition: "all 0.2s ease", marginBottom: 18,
            }}
          >
            <input ref={fileRef} type="file" accept="image/*" style={{ display: "none" }} onChange={(e) => handleFile(e.target.files[0])} />
            {image ? (
              <>
                <img src={image} alt="X-ray" style={{ width: "100%", maxHeight: 200, objectFit: "contain", borderRadius: 8, border: `1px solid ${palette.border}` }} />
                <p style={{ fontSize: 11, color: palette.muted, marginTop: 8 }}>{imageFile?.name} · Click to change</p>
              </>
            ) : (
              <>
                <div style={{ fontSize: 32, marginBottom: 8 }}>🩻</div>
                <p style={{ fontSize: 14, color: palette.navyMid, fontWeight: 600 }}>Drop a chest X-ray here</p>
                <p style={{ fontSize: 12, color: palette.muted, marginTop: 4 }}>or click to browse · PNG / JPG</p>
              </>
            )}
          </div>

          {/* WBC */}
          <div style={{ marginBottom: 16 }}>
            <label style={labelStyle}>WBC Lab Value (K/µL)</label>
            <input style={inputStyle} type="number" step="0.1" value={labValue} onChange={(e) => setLabValue(e.target.value)} />
          </div>

          {/* Reference range */}
          <div style={{ marginBottom: 20 }}>
            <label style={labelStyle}>Reference Range (K/µL)</label>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
              <div>
                <label style={{ ...labelStyle, color: palette.muted }}>Low</label>
                <input style={inputStyleMuted} type="text" step="0.1" value={refLow} disabled={true} onChange={(e) => setRefLow(e.target.value)} />
              </div>
              <div>
                <label style={{ ...labelStyle, color: palette.muted }}>High</label>
                <input style={inputStyleMuted} type="text" step="0.1" value={refHigh} disabled={true} onChange={(e) => setRefHigh(e.target.value)} />
              </div>
            </div>
          </div>

          <button
            onClick={onSubmit}
            disabled={!imageFile || loading}
            style={{
              width: "100%", padding: "13px 20px",
              background: (!imageFile || loading) ? "#94A3B8" : palette.navyMid,
              color: "#fff", border: "none", borderRadius: 8,
              fontSize: 15, fontWeight: 700, cursor: (!imageFile || loading) ? "not-allowed" : "pointer",
              display: "flex", alignItems: "center", justifyContent: "center", gap: 8,
            }}
          >
            {loading ? <><Spinner /> Analysing...</> : "Run Analysis"}
          </button>

          {error && (
            <div style={{ background: "#FEF2F2", border: "1.5px solid #FCA5A5", borderRadius: 8, padding: "12px 16px", color: "#991B1B", fontSize: 13, marginTop: 14 }}>
              ⚠️ {error}
            </div>
          )}
        </div>

        {/* ── Right: Results ── */}
        <div>
          {!result ? (
            <div style={{ ...cardStyle, textAlign: "center", padding: "60px 20px", color: palette.muted }}>
              <div style={{ fontSize: 40, marginBottom: 14, opacity: 0.35 }}>📋</div>
              <p style={{ fontSize: 14, lineHeight: 1.7 }}>
                Upload a chest X-ray and enter a lab value,<br />
                then click <strong>Run Analysis</strong> to see results.
              </p>
            </div>
          ) : (
            <div className="fade" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 18 }}>

              {/* ── Conflict & Defer banners (full width) ── */}
              <div style={{ gridColumn: "1 / -1", display: "grid", gridTemplateColumns: "1fr 1fr", gap: 18 }}>
                {/* Conflict */}
                <div style={{
                  background: conflictDetected ? palette.conflictBg : "#F0FDF4",
                  border: `1.5px solid ${conflictDetected ? palette.conflict : "#86EFAC"}`,
                  borderRadius: 10, padding: "14px 18px", display: "flex", alignItems: "center", gap: 10,
                }}>
                  <span style={{ fontSize: 22 }}>{conflictDetected ? "⚠️" : "✅"}</span>
                  <div>
                    <p style={{ fontSize: 14, fontWeight: 700, color: conflictDetected ? palette.conflict : "#15803D" }}>
                      {conflictDetected ? "Conflict Detected" : "No Conflict"}
                    </p>
                    <p style={{ fontSize: 12, color: palette.muted, marginTop: 2 }}>
                      {conflictDetected ? "Image-only and multimodal predictions disagree." : "Both predictions are consistent."}
                    </p>
                  </div>
                </div>

                {/* Defer gate */}
                <div style={{
                  background: deferGate ? "#FEF2F2" : "#F0FDF4",
                  border: `1.5px solid ${deferGate ? "#FCA5A5" : "#86EFAC"}`,
                  borderRadius: 10, padding: "14px 18px", display: "flex", alignItems: "center", gap: 10,
                }}>
                  <span style={{ fontSize: 22 }}>{deferGate ? "🔴" : "🟢"}</span>
                  <div>
                    <p style={{ fontSize: 14, fontWeight: 700, color: deferGate ? palette.defer : palette.proceed }}>
                      Defer Gate: {result.defer_gate}
                    </p>
                    <p style={{ fontSize: 12, color: palette.muted, marginTop: 2 }}>
                      {deferGate ? "Flagged for clinical review — do not rely on AI alone." : "Predictions are consistent — safe to proceed."}
                    </p>
                  </div>
                </div>
              </div>

              {/* ── Prediction summary ── */}
              <div style={cardStyle}>
                <SectionTitle>Prediction Summary</SectionTitle>
                <ResultRow label="Image-Only Prediction">
                  <Badge bg={predictionColor(result.image_only_prediction)[0]} color={predictionColor(result.image_only_prediction)[1]}>
                    {result.image_only_prediction}
                  </Badge>
                </ResultRow>
                <ResultRow label="Multimodal Prediction">
                  <Badge bg={predictionColor(result.multimodal_prediction)[0]} color={predictionColor(result.multimodal_prediction)[1]}>
                    {result.multimodal_prediction}
                  </Badge>
                </ResultRow>
                <ResultRow label="WBC Lab Value">
                  <span style={{ fontSize: 14, fontWeight: 700, color: labColor(result.lab_status) }}>
                    {result.lab_value} K/µL
                  </span>
                </ResultRow>
                <ResultRow label="Lab Status" noBorder>
                  <Badge bg={labBg(result.lab_status)} color={labColor(result.lab_status)}>
                    {result.lab_status?.toUpperCase()}
                  </Badge>
                </ResultRow>
              </div>

              {/* ── Image Classification ── */}
              <div style={cardStyle}>
                <SectionTitle>Image Classification</SectionTitle>
                <ResultRow label="Detected Type">
                  <Badge bg={imageTypeColor(result.image_type)[0]} color={imageTypeColor(result.image_type)[1]}>
                    {result.image_type}
                  </Badge>
                </ResultRow>
                {/* <ResultRow label="Model" noBorder>
                  <span style={{ fontSize: 12, color: palette.muted, textAlign: "right", maxWidth: 180 }}>
                    {result.model?.name}
                  </span>
                </ResultRow> */}
                <div style={{ marginTop: 14 }}>
                  <p style={{ fontSize: 12, fontWeight: 600, color: palette.muted, marginBottom: 10, letterSpacing: "0.3px" }}>CONFIDENCE SCORES</p>
                  {confidenceScores.map(([label, val]) => (
                    <div key={label} style={{ display: "flex", alignItems: "center", marginBottom: 8 }}>
                      <span style={{ fontSize: 12, color: palette.navy, width: 120, flexShrink: 0 }}>{label}</span>
                      <ProgressBar value={val} color={imageTypeColor(label)[1]} />
                      <span style={{ fontSize: 12, fontWeight: 600, color: palette.navy, marginLeft: 10, width: 40, textAlign: "right" }}>
                        {(val * 100).toFixed(0)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              {/* ── Pathology Scores (full width) ── */}
              {pathologyScores.length > 0 && (
                <div style={{ ...cardStyle, gridColumn: "1 / -1" }}>
                  <SectionTitle>Pathology Scores</SectionTitle>
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "8px 24px" }}>
                    {pathologyScores.map(([name, score]) => {
                      const pct = score * 100;
                      const color = pct >= 60 ? "#DC2626" : pct >= 40 ? "#D97706" : "#16A34A";
                      return (
                        <div key={name} style={{ display: "flex", alignItems: "center", padding: "6px 0", borderBottom: `1px solid ${palette.border}` }}>
                          <span style={{ fontSize: 12, color: palette.navy, width: 160, flexShrink: 0 }}>{name}</span>
                          <ProgressBar value={score} color={color} />
                          <span style={{ fontSize: 12, fontWeight: 700, color, marginLeft: 10, width: 36, textAlign: "right" }}>
                            {pct.toFixed(0)}%
                          </span>
                        </div>
                      );
                    })}
                  </div>
                  <p style={{ fontSize: 11, color: palette.muted, marginTop: 12 }}>
                    🔴 ≥ 60% · 🟡 40–59% · 🟢 &lt; 40% &nbsp;·&nbsp; Composite score: <strong>{(result.model?.composite_score * 100).toFixed(1)}%</strong>
                  </p>
                </div>
              )}

            </div>
          )}
        </div>
      </div>
    </div>
  );
}