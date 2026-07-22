import type { PerceptionEvent } from "../types/epp.gen";

/**
 * Id sintético estable para una fila de `events_buffer` (el backend no
 * asigna un id — ver adapter/app.py GET /events). Se usa para sincronizar
 * hover/selección entre PhotoCanvas y EventsTable dentro del mismo poll.
 */
export function eventId(event: PerceptionEvent, index: number): string {
  return `${event.entity_type}:${event.occurred_at}:${event.candidate_ids.join(",")}:${index}`;
}

/** bbox = [x1, y1, x2, y2] en píxeles de la imagen original (sin escalar). */
export function eventBbox(event: PerceptionEvent): number[] | null {
  const payload = event.payload as { bbox?: number[] | null };
  return payload && Array.isArray(payload.bbox) && payload.bbox.length >= 4
    ? payload.bbox
    : null;
}
