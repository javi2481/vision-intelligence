// CapabilityPanel — dual controls: visible (F1 event-gated) + active (F2 server).
//
// visible: client-only; disabled when entity has no events ("sin detecciones").
// active: PUT /capabilities; enabled whenever available; vehicle locked.
import type { CapabilityEntry } from "../api/client";
import type { PerceptionEvent } from "../types/epp.gen";

interface CapabilityDef {
  entityType: string;
  label: string;
}

const GROUPS: { title: string; items: CapabilityDef[] }[] = [
  {
    title: "Base",
    items: [
      { entityType: "vehicle", label: "Vehículos" },
      { entityType: "object", label: "Objetos" },
      { entityType: "face", label: "Caras" },
    ],
  },
  {
    title: "Extendida",
    items: [
      { entityType: "scene", label: "Escena" },
      { entityType: "pose", label: "Pose" },
      { entityType: "text", label: "Texto" },
      { entityType: "face_id", label: "Identidad" },
    ],
  },
  {
    title: "Experimental",
    items: [
      { entityType: "sign", label: "Señales" },
      { entityType: "scene_cls", label: "Clasif. escena" },
      { entityType: "instance", label: "Instancia" },
      { entityType: "small_object", label: "Objeto pequeño" },
      { entityType: "anomaly", label: "Anomalía" },
      { entityType: "open_vocab", label: "Vocabulario abierto" },
    ],
  },
];

interface Props {
  events: PerceptionEvent[];
  visibility: Record<string, boolean>;
  catalog: Record<string, CapabilityEntry>;
  onToggleVisible: (entityType: string, visible: boolean) => void;
  onToggleActive: (entityType: string, active: boolean) => void;
}

export function CapabilityPanel({
  events,
  visibility,
  catalog,
  onToggleVisible,
  onToggleActive,
}: Props) {
  const present = new Set(events.map((e) => e.entity_type));

  return (
    <div className="vi-capability-panel">
      {GROUPS.map((group) => (
        <div className="vi-capability-group" key={group.title}>
          <div className="vi-capability-group-title">{group.title}</div>
          {group.items.map((item) => {
            const hasEvents = present.has(item.entityType);
            const visibleChecked = visibility[item.entityType] !== false;
            const entry = catalog[item.entityType];
            const available = entry?.available === true;
            const activeChecked = entry?.active === true;
            const critical = entry?.critical === true || item.entityType === "vehicle";
            const activeLocked = critical || !available;
            return (
              <div
                key={item.entityType}
                className={`vi-capability-item${!hasEvents && !available ? " vi-capability-disabled" : ""}`}
              >
                <span className="vi-capability-label">{item.label}</span>
                <label
                  className="vi-capability-toggle"
                  title={hasEvents ? "Visible" : "sin detecciones"}
                >
                  <input
                    type="checkbox"
                    checked={hasEvents && visibleChecked}
                    disabled={!hasEvents}
                    onChange={(e) => onToggleVisible(item.entityType, e.target.checked)}
                    aria-label={`${item.label} visible`}
                  />
                  <span>vis</span>
                </label>
                <label
                  className={`vi-capability-toggle${activeLocked ? " vi-capability-disabled" : ""}`}
                  title={
                    critical
                      ? "Vehicle siempre activo"
                      : available
                        ? "Inferencia activa"
                        : "No disponible en este deploy"
                  }
                >
                  <input
                    type="checkbox"
                    checked={activeChecked}
                    disabled={activeLocked}
                    onChange={(e) => onToggleActive(item.entityType, e.target.checked)}
                    aria-label={`${item.label} active`}
                  />
                  <span>act</span>
                </label>
                {!hasEvents && (
                  <em className="vi-capability-empty-hint">sin detecciones</em>
                )}
              </div>
            );
          })}
        </div>
      ))}
    </div>
  );
}
