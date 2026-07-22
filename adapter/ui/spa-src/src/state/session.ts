// Máquina de estados de sesión (Fase 1, addendum-s2-spa-s3).
//
// idle → uploading → processing → ready | degraded | empty | error
//
// Completitud (Claude correction — NO existe `analysis_complete` en el
// envelope): el análisis está completo IFF
//   generation === last_ingest_generation
// (ver GET /events en adapter/app.py). Mientras no sea igual, el estado es
// "processing" — se hace polling a /events cada ~1s. Timeout de 60s sin
// completar → "error" con posibilidad de retry. Los toggles de visibilidad
// (CapabilityPanel) se resetean cuando `generation` cambia — los consumidores
// de este hook deben usar `generation` como dependencia de ese reset.
import { useCallback, useEffect, useRef, useState } from "react";
import {
  clearMedia,
  getCurrentMedia,
  getEvents,
  uploadMedia,
  type EventsEnvelope,
} from "../api/client";
import type { PerceptionEvent } from "../types/epp.gen";

export type SessionStatus =
  | "idle"
  | "uploading"
  | "processing"
  | "ready"
  | "degraded"
  | "empty"
  | "error";

export interface SessionState {
  status: SessionStatus;
  mediaName: string | null;
  generation: number;
  lastIngestGeneration: number | null;
  events: PerceptionEvent[];
  errorMessage: string | null;
}

const POLL_INTERVAL_MS = 1000;
const PROCESSING_TIMEOUT_MS = 60_000;

const initialState: SessionState = {
  status: "idle",
  mediaName: null,
  generation: 0,
  lastIngestGeneration: null,
  events: [],
  errorMessage: null,
};

function deriveStatus(envelope: EventsEnvelope): SessionStatus {
  if (envelope.degraded) return "degraded";
  const complete = envelope.generation === envelope.last_ingest_generation;
  if (!complete) return "processing";
  return envelope.events.length > 0 ? "ready" : "empty";
}

export function useSession() {
  const [state, setState] = useState<SessionState>(initialState);
  const pollTimer = useRef<ReturnType<typeof setInterval> | null>(null);
  const processingSinceRef = useRef<number | null>(null);
  const generationRef = useRef<number>(0);

  const stopPolling = useCallback(() => {
    if (pollTimer.current !== null) {
      clearInterval(pollTimer.current);
      pollTimer.current = null;
    }
  }, []);

  const pollOnce = useCallback(async () => {
    try {
      const envelope = await getEvents();

      // Cambio de generación (p.ej. media watcher auto-seleccionó otra foto):
      // reinicia el cronómetro de timeout. Los consumidores resetean sus
      // propios toggles observando `generation`.
      if (envelope.generation !== generationRef.current) {
        generationRef.current = envelope.generation;
        processingSinceRef.current = Date.now();
      }

      const status = deriveStatus(envelope);
      if (status === "processing" && processingSinceRef.current === null) {
        processingSinceRef.current = Date.now();
      }

      if (
        status === "processing" &&
        processingSinceRef.current !== null &&
        Date.now() - processingSinceRef.current > PROCESSING_TIMEOUT_MS
      ) {
        stopPolling();
        setState((prev) => ({
          ...prev,
          status: "error",
          errorMessage: "Timeout esperando análisis (60s). Reintentar.",
          generation: envelope.generation,
          lastIngestGeneration: envelope.last_ingest_generation,
          events: envelope.events,
        }));
        return;
      }

      if (status !== "processing") {
        processingSinceRef.current = null;
      }

      setState((prev) => ({
        ...prev,
        status,
        generation: envelope.generation,
        lastIngestGeneration: envelope.last_ingest_generation,
        events: envelope.events,
        errorMessage: null,
      }));
    } catch (err) {
      stopPolling();
      setState((prev) => ({
        ...prev,
        status: "error",
        errorMessage: err instanceof Error ? err.message : "Error de red",
      }));
    }
  }, [stopPolling]);

  const startPolling = useCallback(() => {
    stopPolling();
    processingSinceRef.current = Date.now();
    void pollOnce();
    pollTimer.current = setInterval(() => void pollOnce(), POLL_INTERVAL_MS);
  }, [pollOnce, stopPolling]);

  // Bootstrap: si ya hay una foto activa (watcher del adapter la
  // auto-selecciona), arrancar en "processing" y empezar a sondear.
  useEffect(() => {
    void (async () => {
      try {
        const current = await getCurrentMedia();
        generationRef.current = current.generation;
        if (current.name) {
          setState((prev) => ({
            ...prev,
            status: "processing",
            mediaName: current.name,
            generation: current.generation,
          }));
          startPolling();
        }
      } catch {
        // Sin conexión inicial: se queda en idle: el usuario puede subir.
      }
    })();
    return () => stopPolling();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const upload = useCallback(
    async (file: File) => {
      setState((prev) => ({ ...prev, status: "uploading", errorMessage: null }));
      try {
        const res = await uploadMedia(file);
        if (!res.ok || res.generation === undefined) {
          setState((prev) => ({
            ...prev,
            status: "error",
            errorMessage: res.error || res.msg || "Upload falló",
          }));
          return;
        }
        generationRef.current = res.generation;
        setState((prev) => ({
          ...prev,
          status: "processing",
          mediaName: res.name ?? null,
          generation: res.generation ?? prev.generation,
          events: [],
        }));
        startPolling();
      } catch (err) {
        setState((prev) => ({
          ...prev,
          status: "error",
          errorMessage: err instanceof Error ? err.message : "Error de red",
        }));
      }
    },
    [startPolling],
  );

  const clear = useCallback(async () => {
    stopPolling();
    try {
      const res = await clearMedia();
      generationRef.current = res.generation;
      processingSinceRef.current = null;
      setState({ ...initialState, generation: res.generation });
    } catch (err) {
      setState((prev) => ({
        ...prev,
        status: "error",
        errorMessage: err instanceof Error ? err.message : "Error de red",
      }));
    }
  }, [stopPolling]);

  const retry = useCallback(() => {
    setState((prev) => ({ ...prev, status: "processing", errorMessage: null }));
    startPolling();
  }, [startPolling]);

  return { state, upload, clear, retry };
}
