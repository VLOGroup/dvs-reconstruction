# DVS Reconstruction
This repository (will soon) provide software to our publication "Real-Time Intensity-Image Reconstruction for Event Cameras Using Manifold Regularisation", BMVC 2016.

If you use this code please cite the following publication:
~~~
@inproceedings{reinbacher_bmvc2016,
  author = {Christian Reinbacher and Gottfried Graber and Thomas Pock},
  title = {{Real-Time Intensity-Image Reconstruction for Event Cameras Using Manifold Regularisation}},
  booktitle = {2016 British Machine Vision Conference (BMVC)},
  year = {2016},
}
~~~

## Compiling
For your convenience, the required libraries that are on Github are added as
submodules. So clone this repository with `--recursive` or do a
~~~
git submodule update --init --recursive
~~~
after cloning.

To compile this code, make sure you first install ImageUtilities with the `iugui`, `iuio` and `iumath` module. Furthermore, this software requires:
 - GCC >= 4.9
 - CMake >= 3.2
 - Qt >= 5.6
 - Boost (program_options, filesystem, system)
 - libcaer >=2.0 (https://github.com/inilabs/libcaer)
 - cnpy (https://github.com/rogersce/cnpy)
 - DVS128 or DAVIS240 camera (can also load events from files)

 To compile the GUI:
 ~~~
cd cnpy
cmake .
make
(sudo) make install
cd ../libcaer
cmake .
make
(sudo) make install
cd ..
mkdir build
cd build
cmake ..
make -j6
 ~~~

 Per default, the application will compile to support the iniLabs DVS128. If you want to attach a DAVIS240 instead, set the CMake option `WITH_DAVIS`.

## Usage
Launch `live_reconstruction_gui` to get to the main application which should look like this:
<img src="https://github.com/VLOGroup/dvs-reconstruction/raw/master/images/screenshot.png"></img>
Clicking on the play button with an attached camera will start the live reconstruction method. Alternatively, events can be loaded from text files with one event per line:
~~~
<timestamp in seconds> <x> <y> <polarity (-1/1)>
...
...
~~~
