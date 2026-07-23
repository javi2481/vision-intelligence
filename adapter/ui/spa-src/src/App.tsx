// App — shell de la SPA Fase 1+2 (dual visible/active capabilities).
import { useCallback, useEffect, useState } from "react";
import {
  getCapabilities,
  putCapabilities,
  type CapabilityEntry,
} from "./api/client";
import { AnalyticsRow } from "./components/AnalyticsRow";
import { CapabilityPanel } from "./components/CapabilityPanel";
import { EventsTable } from "./components/EventsTable";
import { PhotoCanvas } from "./components/PhotoCanvas";
import { UploadBar } from "./components/UploadBar";
import { useSession } from "./state/session";

export function App() {
  const { state, upload, clear, retry } = useSession();
  const [visibility, setVisibility] = useState<Record<string, boolean>>({});
  const [catalog, setCatalog] = useState<Record<string, CapabilityEntry>>({});
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const refreshCatalog = useCallback(async () => {
    try {
      const res = await getCapabilities();
      setCatalog(res.capabilities);
    } catch {
      // Adapter offline: keep last catalog.
    }
  }, []);

  useEffect(() => {
    void refreshCatalog();
  }, [refreshCatalog]);

  // F1: reset visibility on generation bump. Do NOT reset server active.
  useEffect(() => {
    setVisibility({});
    setHoveredId(null);
    setSelectedId(null);
  }, [state.generation]);

  // After generation bump (upload or PUT), re-sync active from server.
  useEffect(() => {
    void refreshCatalog();
  }, [state.generation, refreshCatalog]);

  const onToggleActive = useCallback(
    async (entityType: string, active: boolean) => {
      try {
        const res = await putCapabilities({ [entityType]: active });
        setCatalog(res.capabilities);
      } catch (err) {
        console.error(err);
        void refreshCatalog();
      }
    },
    [refreshCatalog],
  );

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
            catalog={catalog}
            onToggleVisible={(entityType, visible) =>
              setVisibility((prev) => ({ ...prev, [entityType]: visible }))
            }
            onToggleActive={(entityType, active) =>
              void onToggleActive(entityType, active)
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
