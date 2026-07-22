// PhotoCanvas — foto original + overlay SVG (Fase 1, addendum-s2-spa-s3).
//
// REQ-S2: viewBox = naturalWidth×naturalHeight; los bbox del payload ya
// están en píxeles de la imagen ORIGINAL (`draw_preview` en
// detection/common/preview.py dibuja sobre el mismo frame que
// GET /media/original) — sin escala manual, el <svg> escala solo junto al
// <img> porque comparten el mismo contenedor y viewBox.
import { useState } from "react";
import { originalMediaUrl } from "../api/client";
import { colorForEvent } from "../colors/entityColors.gen";
import type { PerceptionEvent } from "../types/epp.gen";
import { eventBbox, eventId } from "../utils/eventId";

interface Props {
  generation: number;
  events: PerceptionEvent[];
  visibility: Record<string, boolean>;
  hoveredId: string | null;
  selectedId: string | null;
  onHover: (id: string | null) => void;
  onSelect: (id: string | null) => void;
}

export function PhotoCanvas({
  generation,
  events,
  visibility,
  hoveredId,
  selectedId,
  onHover,
  onSelect,
}: Props) {
  const [naturalSize, setNaturalSize] = useState<{ w: number; h: number } | null>(null);

  const visibleEntries = events
    .map((event, index) => ({ event, index, id: eventId(event, index), bbox: eventBbox(event) }))
    .filter((entry) => entry.bbox !== null && visibility[entry.event.entity_type] !== false);

  const selected = visibleEntries.find((entry) => entry.id === selectedId) ?? null;

  return (
    <div className="vi-canvas">
      <img
        key={generation}
        className="vi-canvas-img"
        src={originalMediaUrl(generation)}
        alt="Foto activa"
        onLoad={(e) => {
          const img = e.currentTarget;
          setNaturalSize({ w: img.naturalWidth, h: img.naturalHeight });
        }}
      />
      {naturalSize && (
        <svg
          className="vi-canvas-overlay"
          viewBox={`0 0 ${naturalSize.w} ${naturalSize.h}`}
          preserveAspectRatio="xMidYMid meet"
        >
          {visibleEntries.map(({ event, id, bbox }) => {
            const [x1, y1, x2, y2] = bbox as number[];
            const color = colorForEvent(
              event.entity_type,
              (event.payload as { vehicle_type?: string | null }).vehicle_type,
              event.candidate_ids[0],
            );
            const isHovered = id === hoveredId;
            const isSelected = id === selectedId;
            return (
              <rect
                key={id}
                x={Math.min(x1, x2)}
                y={Math.min(y1, y2)}
                width={Math.abs(x2 - x1)}
                height={Math.abs(y2 - y1)}
                fill={isSelected ? color : "transparent"}
                fillOpacity={isSelected ? 0.15 : 0}
                stroke={color}
                strokeWidth={isHovered || isSelected ? 4 : 2}
                style={{ cursor: "pointer" }}
                onMouseEnter={() => onHover(id)}
                onMouseLeave={() => onHover(null)}
                onClick={() => onSelect(id === selectedId ? null : id)}
              />
            );
          })}
        </svg>
      )}
      {selected && naturalSize && (
        <EventPopover
          event={selected.event}
          bbox={selected.bbox as number[]}
          naturalSize={naturalSize}
          onClose={() => onSelect(null)}
        />
      )}
      {!naturalSize && <div className="vi-canvas-loading">Cargando foto…</div>}
    </div>
  );
}

function EventPopover({
  event,
  bbox,
  naturalSize,
  onClose,
}: {
  event: PerceptionEvent;
  bbox: number[];
  naturalSize: { w: number; h: number };
  onClose: () => void;
}) {
  const [x1, , , y2] = bbox;
  // Posición en % del contenedor (no píxeles) — se mantiene alineada al
  // bbox sin importar el zoom/tamaño real de la imagen renderizada.
  const left = `${(Math.max(0, x1) / naturalSize.w) * 100}%`;
  const top = `${(Math.min(naturalSize.h, y2) / naturalSize.h) * 100}%`;
  const payload = event.payload as unknown as Record<string, unknown>;

  return (
    <div className="vi-popover" style={{ left, top }}>
      <button className="vi-popover-close" onClick={onClose} aria-label="Cerrar">
        ×
      </button>
      <div className="vi-popover-title">{event.entity_type}</div>
      <div className="vi-popover-row">confidence: {event.confidence.toFixed(2)}</div>
      {Object.entries(payload)
        .filter(([k, v]) => k !== "bbox" && v !== null && v !== undefined)
        .map(([k, v]) => (
          <div className="vi-popover-row" key={k}>
            {k}: {typeof v === "object" ? JSON.stringify(v) : String(v)}
          </div>
        ))}
    </div>
  );
}
