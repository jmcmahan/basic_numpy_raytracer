# Basic Numpy Raytracer
This is a basic raytracer written in Python. It uses Numpy to do most of the calculations in arrays instead of loops for efficiency. The "R03_..." script calls the raytracer to implement the scene that was rendered to this [video](https://www.youtube.com/watch?v=WY2jIqbtUDk) (though I accidentally uploaded a 30 fps version rather than the 60 fps version the script was intended for). The script either writes out image frames or try to display them using the included functions in "graphics_toolbox.py" which use the Pygame library for display. There's a bash script with the ffmpeg command I used for rendering the final video.

## Resources
The raytracing code is based on Dmitry V. Sokolov's [tinyraytracer](https://github.com/ssloy/tinyraytracer/wiki/Part-1:-understandable-raytracing) tutorial. 

The graphics display code in "graphics_toolbox.py" is based on the examples from Karthik Karanth's [blog post](https://karthikkaranth.me/blog/drawing-pixels-with-python/).

The snow texture is from a website called [Spiral Graphics](http://spiralgraphics.biz/packs/snow_ice/index.htm?23).
