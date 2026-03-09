import { Loader2, AlertTriangle } from "lucide-react";

interface PageLoaderProps {
  message?: string;
}

export function PageLoader({ message = "Loading..." }: PageLoaderProps) {
  return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", gap: 10 }}>
      <Loader2 size={20} style={{ color: "var(--accent)", animation: "spin 1s linear infinite" }} />
      <span style={{ fontSize: 13, color: "var(--text-muted)" }}>{message}</span>
    </div>
  );
}

interface PageErrorProps {
  message?: string;
  onBack?: () => void;
  backLabel?: string;
}

export function PageError({ message = "Something went wrong", onBack, backLabel = "Go back" }: PageErrorProps) {
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: "100%", gap: 12 }}>
      <AlertTriangle size={24} style={{ color: "var(--red)" }} />
      <span style={{ fontSize: 13, color: "var(--text-muted)" }}>{message}</span>
      {onBack && (
        <button
          onClick={onBack}
          style={{
            padding: "6px 14px", borderRadius: 7,
            border: "1px solid var(--border)", background: "var(--surface)",
            fontSize: 12, color: "var(--text-secondary)", cursor: "pointer",
          }}
        >
          {backLabel}
        </button>
      )}
    </div>
  );
}
