const STEPS = [
  { label: "OCR", icon: "🔍" },
  { label: "Clean", icon: "🧹" },
  { label: "Normalize", icon: "✨" },
];

export default function StepsIndicator({ activeStep }) {
  return (
    <div className="steps">
      {STEPS.map((s, i) => {
        const state =
          activeStep === null ? "" :
          i < activeStep ? "done" :
          i === activeStep ? "active" : "";
        return (
          <div key={s.label} className={`step ${state}`}>
            <div className="step-dot">
              {state === "done" ? "✓" : s.icon}
            </div>
            <div className="step-label">{s.label}</div>
          </div>
        );
      })}
    </div>
  );
}
