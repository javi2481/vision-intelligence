// Wrapper mínimo de ECharts — sin CDN en runtime (bundle vía npm, ver
// package.json). Un <div> + una instancia por componente montado.
//
// Import selectivo (echarts/core + solo los charts/componentes que usa
// AnalyticsRow: pie/bar) en vez de `import * as echarts from "echarts"` —
// reduce el bundle de ~1.2MB a una fracción sin perder funcionalidad.
import * as echarts from "echarts/core";
import { BarChart, PieChart } from "echarts/charts";
import { GridComponent, TitleComponent, TooltipComponent } from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";
import { useEffect, useRef } from "react";

echarts.use([PieChart, BarChart, GridComponent, TitleComponent, TooltipComponent, CanvasRenderer]);

interface Props {
  option: echarts.EChartsCoreOption;
  height?: number;
}

export function EChart({ option, height = 220 }: Props) {
  const ref = useRef<HTMLDivElement | null>(null);
  const instanceRef = useRef<echarts.ECharts | null>(null);

  useEffect(() => {
    if (!ref.current) return;
    const instance = echarts.init(ref.current);
    instanceRef.current = instance;
    const onResize = () => instance.resize();
    window.addEventListener("resize", onResize);
    return () => {
      window.removeEventListener("resize", onResize);
      instance.dispose();
      instanceRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    instanceRef.current?.setOption(option, true);
  }, [option]);

  return <div ref={ref} style={{ width: "100%", height }} />;
}
