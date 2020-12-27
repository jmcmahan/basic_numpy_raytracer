ffmpeg -r 60 -i image%03d.png -c:v libx264  -pix_fmt rgb24 out.mp4
