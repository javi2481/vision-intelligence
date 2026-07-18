#!/bin/sh
exec paddlex --serve --pipeline vehicle_attribute_recognition --port 8080 --device "${VI_DEVICE:-cpu}"
