// EventsTable — tabla colapsable de PerceptionEvent visibles, sincronizada
// con el hover/selección de PhotoCanvas vía el mismo `eventId`.
import { useState } from "react";
import type { PerceptionEvent } from "../types/epp.gen";
import { eventId } from "../utils/eventId";

interface Props {
  events: PerceptionEvent[];
  visibility: Record<string, boolean>;
  hoveredId: string | null;
  selectedId: string | null;
  onHover: (id: string | null) => void;
  onSelect: (id: string | null) => void;
}

function payloadSummary(event: PerceptionEvent): string {
  const p = event.payload as unknown as Record<string, unknown>;
  const parts: string[] = [];
  if (p.vehicle_type) parts.push(String(p.vehicle_type));
  if (p.color) parts.push(String(p.color));
  if (p.plate_text) parts.push(`plate:${p.plate_text}`);
  if (p.class_name) parts.push(String(p.class_name));
  if (p.text) parts.push(String(p.text).slice(0, 24));
  if (p.identity) parts.push(`id:${p.identity}`);
  return parts.join(" · ") || "—";
}

export function EventsTable({ events, visibility, hoveredId, selectedId, onHover, onSelect }: Props) {
  const [collapsed, setCollapsed] = useState(false);

  const rows = events
    .map((event, index) => ({ event, id: eventId(event, index) }))
    .filter(({ event }) => visibility[event.entity_type] !== false);

  return (
    <div className="vi-events-table">
      <button className="vi-collapse-toggle" onClick={() => setCollapsed((v) => !v)}>
        {collapsed ? "▶" : "▼"} Eventos ({rows.length})
      </button>
      {!collapsed && (
        <table className="vi-table">
          <thead>
            <tr>
              <th>Tipo</th>
              <th>Confianza</th>
              <th>Detalle</th>
              <th>Hora</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(({ event, id }) => (
              <tr
                key={id}
                className={
                  id === selectedId ? "vi-row-selected" : id === hoveredId ? "vi-row-hovered" : ""
                }
                onMouseEnter={() => onHover(id)}
                onMouseLeave={() => onHover(null)}
                onClick={() => onSelect(id === selectedId ? null : id)}
              >
                <td>
                  <span className="vi-pill vi-pill-type">{event.entity_type}</span>
                </td>
                <td>{event.confidence.toFixed(2)}</td>
                <td>{payloadSummary(event)}</td>
                <td>{new Date(event.occurred_at).toLocaleTimeString()}</td>
              </tr>
            ))}
            {rows.length === 0 && (
              <tr>
                <td colSpan={4} className="vi-muted">
                  Sin eventos visibles.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      )}
    </div>
  );
}
