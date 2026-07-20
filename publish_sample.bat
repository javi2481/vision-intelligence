@echo off
REM Publica un video de muestra en loop hacia MediaMTX (camino "vivo" reproducible).
REM Ejecutar en el HOST con Docker Desktop + compose ya levantados (mediamtx up).
REM
REM Uso:
REM   publish_sample.bat
REM   publish_sample.bat videos_muestra\Brasil6.mp4
REM
REM Luego en el panel: "Limpiar selección (volver a RTSP vivo)" para que el bridge
REM lea rtsp://mediamtx:8554/webcam en vez de un archivo local.

setlocal EnableExtensions
set "RTSP_URL=rtsp://localhost:8554/webcam"
set "SAMPLE=%~1"
if "%SAMPLE%"=="" set "SAMPLE=videos_muestra\Brasil6.mp4"

if not exist "%SAMPLE%" (
  echo ERROR: no existe "%SAMPLE%"
  echo Coloca el archivo bajo videos_muestra\ o pasa la ruta como argumento.
  exit /b 1
)

echo ==> Publicando "%SAMPLE%" en loop -^> %RTSP_URL%
echo     En el panel: limpiar seleccion de muestra local.
echo     Ctrl+C para detener.

ffmpeg -hide_banner -loglevel info ^
  -re -stream_loop -1 -i "%SAMPLE%" ^
  -c:v libx264 -preset ultrafast -tune zerolatency -pix_fmt yuv420p -an ^
  -f rtsp -rtsp_transport tcp ^
  %RTSP_URL%

endlocal
