// UploadBar — subir foto, limpiar, y status legible (idle/processing/…).
import { useRef } from "react";
import type { SessionStatus } from "../state/session";

const STATUS_LABEL: Record<SessionStatus, string> = {
  idle: "Sin foto — subí una imagen",
  uploading: "Subiendo…",
  processing: "Procesando…",
  ready: "Listo",
  degraded: "Degradado (PaddleX no disponible)",
  empty: "Completo — sin detecciones",
  error: "Error",
};

interface Props {
  status: SessionStatus;
  errorMessage: string | null;
  onUpload: (file: File) => void;
  onClear: () => void;
  onRetry: () => void;
}

export function UploadBar({ status, errorMessage, onUpload, onClear, onRetry }: Props) {
  const inputRef = useRef<HTMLInputElement | null>(null);

  return (
    <div className="vi-upload-bar">
      <input
        ref={inputRef}
        type="file"
        accept=".jpg,.jpeg,.png,.bmp,image/jpeg,image/png,image/bmp"
        style={{ display: "none" }}
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) onUpload(file);
          e.target.value = "";
        }}
      />
      <button
        className="vi-btn vi-btn-primary"
        disabled={status === "uploading"}
        onClick={() => inputRef.current?.click()}
      >
        Subir foto
      </button>
      <button className="vi-btn" onClick={onClear} disabled={status === "idle"}>
        Limpiar
      </button>
      {status === "error" && (
        <button className="vi-btn" onClick={onRetry}>
          Reintentar
        </button>
      )}
      <span className={`vi-status-pill vi-status-${status}`}>
        {STATUS_LABEL[status]}
        {status === "error" && errorMessage ? `: ${errorMessage}` : ""}
      </span>
    </div>
  );
}
