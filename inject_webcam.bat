@echo off
REM Inyecta la webcam del host (Windows DirectShow) hacia MediaMTX.
REM Ejecutar en el HOST con Docker Desktop + compose ya levantados.
REM
REM 1) Listar dispositivos:
REM      ffmpeg -list_devices true -f dshow -i dummy
REM 2) Copiar el nombre exacto de la cámara (ej. "Integrated Camera")
REM 3) Editar WEBCAM_NAME abajo o pasar como argumento:
REM      inject_webcam.bat "Integrated Camera"

setlocal EnableExtensions
set "RTSP_URL=rtsp://localhost:8554/webcam"
set "WEBCAM_NAME=%~1"
if "%WEBCAM_NAME%"=="" set "WEBCAM_NAME=Integrated Camera"

echo ==> Listar webcams (si falla la inyeccion):
echo     ffmpeg -list_devices true -f dshow -i dummy
echo ==> Inyectando video="%WEBCAM_NAME%" -^> %RTSP_URL%
echo     Ctrl+C para detener.

ffmpeg -hide_banner -loglevel info ^
  -f dshow -i video="%WEBCAM_NAME%" ^
  -c:v libx264 -preset ultrafast -tune zerolatency -pix_fmt yuv420p ^
  -f rtsp -rtsp_transport tcp ^
  %RTSP_URL%

endlocal
