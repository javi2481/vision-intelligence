// AnalyticsRow — agregados 100% client-side sobre los eventos VISIBLES
// (Fase 1, addendum-s2-spa-s3). Nada de esto pega al backend: se recalcula
// en cada render a partir del array de eventos ya filtrado por
// CapabilityPanel — coherente con "lo que ves es lo que se analiza".
import { useMemo } from "react";
import { colorForEvent, VEHICLE_TYPE_COLORS } from "../colors/entityColors.gen";
import type { PerceptionEvent } from "../types/epp.gen";
import { EChart } from "./EChart";

interface Props {
  events: PerceptionEvent[];
  visibility: Record<string, boolean>;
}

function isVisible(event: PerceptionEvent, visibility: Record<string, boolean>): boolean {
  return visibility[event.entity_type] !== false;
}

export function AnalyticsRow({ events, visibility }: Props) {
  const visibleEvents = useMemo(
    () => events.filter((e) => isVisible(e, visibility)),
    [events, visibility],
  );

  const vehicleEvents = useMemo(
    () => visibleEvents.filter((e) => e.entity_type === "vehicle"),
    [visibleEvents],
  );

  const vehicleTypeOption = useMemo(() => {
    const counts = new Map<string, number>();
    for (const e of vehicleEvents) {
      const vt = (e.payload as { vehicle_type?: string | null }).vehicle_type || "desconocido";
      counts.set(vt, (counts.get(vt) ?? 0) + 1);
    }
    const data = [...counts.entries()].map(([name, value]) => ({
      name,
      value,
      itemStyle: { color: VEHICLE_TYPE_COLORS[name.toLowerCase()] },
    }));
    return {
      title: { text: "Tipo de vehículo", left: "center", textStyle: { fontSize: 12 } },
      tooltip: { trigger: "item" },
      series: [{ type: "pie", radius: ["35%", "70%"], data }],
    };
  }, [vehicleEvents]);

  const colorOption = useMemo(() => {
    const counts = new Map<string, number>();
    for (const e of vehicleEvents) {
      const color = (e.payload as { color?: string | null }).color || "desconocido";
      counts.set(color, (counts.get(color) ?? 0) + 1);
    }
    const entries = [...counts.entries()];
    return {
      title: { text: "Color", left: "center", textStyle: { fontSize: 12 } },
      tooltip: { trigger: "axis" },
      xAxis: { type: "category", data: entries.map(([k]) => k) },
      yAxis: { type: "value" },
      series: [{ type: "bar", data: entries.map(([, v]) => v) }],
    };
  }, [vehicleEvents]);

  const confidenceOption = useMemo(() => {
    const buckets = new Array(10).fill(0);
    for (const e of visibleEvents) {
      const idx = Math.min(9, Math.max(0, Math.floor(e.confidence * 10)));
      buckets[idx] += 1;
    }
    return {
      title: { text: "Confianza", left: "center", textStyle: { fontSize: 12 } },
      tooltip: { trigger: "axis" },
      xAxis: {
        type: "category",
        data: buckets.map((_, i) => `${i / 10}-${(i + 1) / 10}`),
        axisLabel: { fontSize: 9, rotate: 45 },
      },
      yAxis: { type: "value" },
      series: [
        {
          type: "bar",
          data: buckets,
          itemStyle: { color: colorForEvent("vehicle") },
        },
      ],
    };
  }, [visibleEvents]);

  const stats = useMemo(() => {
    const uniquePlates = new Set(
      vehicleEvents
        .map((e) => (e.payload as { plate_text?: string | null }).plate_text)
        .filter((p): p is string => Boolean(p)),
    );
    const byEntity = new Map<string, number>();
    for (const e of visibleEvents) byEntity.set(e.entity_type, (byEntity.get(e.entity_type) ?? 0) + 1);
    return {
      total: visibleEvents.length,
      uniquePlates: uniquePlates.size,
      byEntity,
    };
  }, [visibleEvents, vehicleEvents]);

  return (
    <div className="vi-analytics-row">
      <div className="vi-analytics-stats">
        <div className="vi-stat">
          <div className="vi-stat-value">{stats.total}</div>
          <div className="vi-stat-label">Eventos visibles</div>
        </div>
        <div className="vi-stat">
          <div className="vi-stat-value">{stats.uniquePlates}</div>
          <div className="vi-stat-label">Patentes únicas</div>
        </div>
        <div className="vi-stat vi-stat-breakdown">
          {[...stats.byEntity.entries()].map(([entity, count]) => (
            <span key={entity} className="vi-pill vi-pill-type">
              {entity}: {count}
            </span>
          ))}
        </div>
      </div>
      <div className="vi-analytics-charts">
        <EChart option={vehicleTypeOption} />
        <EChart option={colorOption} />
        <EChart option={confidenceOption} />
      </div>
    </div>
  );
}
