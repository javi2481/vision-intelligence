// App — shell de la SPA Fase 1 (addendum-s2-spa-s3). Coexiste con AMIS en
// `/` (ver adapter/ui/README.md); esta SPA monta en `/app/`.
import { useEffect, useState } from "react";
import { AnalyticsRow } from "./components/AnalyticsRow";
import { CapabilityPanel } from "./components/CapabilityPanel";
import { EventsTable } from "./components/EventsTable";
import { PhotoCanvas } from "./components/PhotoCanvas";
import { UploadBar } from "./components/UploadBar";
import { useSession } from "./state/session";

export function App() {
  const { state, upload, clear, retry } = useSession();
  const [visibility, setVisibility] = useState<Record<string, boolean>>({});
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  // REQ-S3 / T2.3: resetear toggles y selección cuando la generación activa
  // cambia (nueva foto subida/seleccionada) — evita "arrastrar" filtros de
  // una foto a otra.
  useEffect(() => {
    setVisibility({});
    setHoveredId(null);
    setSelectedId(null);
  }, [state.generation]);

  return (
    <div className="vi-app">
      <header className="vi-header">
        <h1>Vision Intelligence — SPA (Fase 1)</h1>
        <UploadBar
          status={state.status}
          errorMessage={state.errorMessage}
          onUpload={(file) => void upload(file)}
          onClear={() => void clear()}
          onRetry={retry}
        />
      </header>

      <div className="vi-layout">
        <aside className="vi-sidebar">
          <CapabilityPanel
            events={state.events}
            visibility={visibility}
            onToggle={(entityType, visible) =>
              setVisibility((prev) => ({ ...prev, [entityType]: visible }))
            }
          />
        </aside>

        <main className="vi-main">
          {state.status === "idle" ? (
            <div className="vi-empty-card">
              <strong>Sin foto activa</strong>
              <p>Subí una imagen para empezar el análisis.</p>
            </div>
          ) : (
            <PhotoCanvas
              generation={state.generation}
              events={state.events}
              visibility={visibility}
              hoveredId={hoveredId}
              selectedId={selectedId}
              onHover={setHoveredId}
              onSelect={setSelectedId}
            />
          )}
        </main>
      </div>

      <AnalyticsRow events={state.events} visibility={visibility} />

      <EventsTable
        events={state.events}
        visibility={visibility}
        hoveredId={hoveredId}
        selectedId={selectedId}
        onHover={setHoveredId}
        onSelect={setSelectedId}
      />
    </div>
  );
}
