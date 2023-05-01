# Start X virtual frame buffer

`$ Xvfb :99 -screen 0 1024x768x24 &`

# set display to the new frame buffer

`$ export DISPLAY=:99`
