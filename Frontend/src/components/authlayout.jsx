const stages = [
  "Cleaning",
  "EDA",
  "Feature Engineering",
  "Model Selection",
  "Hyperparameter Tuning",
  "Evaluation",
  "Reporting",
];

function AuthLayout({ title, subtitle, children, footer }) {
  return (
    <div className="min-h-screen flex bg-paper">
      {/* Left panel — pipeline diagram, hidden on small screens */}
      <div className="hidden lg:flex lg:w-1/2 bg-ink flex-col justify-center px-16 py-12">
        <p className="font-mono text-sm text-accent mb-10 tracking-tight">
          AI Data Scientist
        </p>
        <div className="space-y-0">
          {stages.map((stage, i) => (
            <div key={stage} className="flex items-start gap-4">
              <div className="flex flex-col items-center">
                <div className="w-8 h-8 rounded-full border border-muted flex items-center justify-center text-xs font-mono text-muted">
                  {i + 1}
                </div>
                {i < stages.length - 1 && (
                  <div className="w-px h-8 bg-panel" />
                )}
              </div>
              <p className="font-mono text-sm text-paper pt-1.5">{stage}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Right panel — the actual form */}
      <div className="w-full lg:w-1/2 flex flex-col justify-center px-8 sm:px-16 py-12">
        <div className="max-w-sm w-full mx-auto">
          <h1 className="font-serif text-3xl text-ink mb-2">{title}</h1>
          <p className="text-muted text-sm mb-8">{subtitle}</p>
          {children}
          {footer && <div className="mt-6 text-sm text-muted">{footer}</div>}
        </div>
      </div>
    </div>
  );
}

export default AuthLayout;