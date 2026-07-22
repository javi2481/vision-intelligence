// CapabilityPanel — toggles de visibilidad por entity_type (Fase 1).
//
// REQ-S3: shape lógico por capacidad es `{ visible, active? }`; `active`
// (inferencia on/off) NO se implementa/ignora en F1 — no hay /capabilities
// todavía (ver hard constraints del addendum). Solo se listan/habilitan
// tipos que YA tienen eventos en el buffer actual; el resto queda
// deshabilitado con "sin detecciones".
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
  onToggle: (entityType: string, visible: boolean) => void;
}

export function CapabilityPanel({ events, visibility, onToggle }: Props) {
  const present = new Set(events.map((e) => e.entity_type));

  return (
    <div className="vi-capability-panel">
      {GROUPS.map((group) => (
        <div className="vi-capability-group" key={group.title}>
          <div className="vi-capability-group-title">{group.title}</div>
          {group.items.map((item) => {
            const hasEvents = present.has(item.entityType);
            const checked = visibility[item.entityType] !== false;
            return (
              <label
                key={item.entityType}
                className={`vi-capability-item${hasEvents ? "" : " vi-capability-disabled"}`}
                title={hasEvents ? undefined : "sin detecciones"}
              >
                <input
                  type="checkbox"
                  checked={hasEvents && checked}
                  disabled={!hasEvents}
                  onChange={(e) => onToggle(item.entityType, e.target.checked)}
                />
                <span>{item.label}</span>
                {!hasEvents && <em className="vi-capability-empty-hint">sin detecciones</em>}
              </label>
            );
          })}
        </div>
      ))}
    </div>
  );
}
