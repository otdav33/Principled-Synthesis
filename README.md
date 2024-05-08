# Principled-Synthesis
A synthesizer that uses sliders to replicate a wide variety of instruments

Dependencies:
- python
- python packages: tk librosa numpy scipy matplotlib pygame
- polyphone for automatically converting .sfz soundfonts to .sf2 (it will run without it, but will spit out an error that you can ignore)

## Features

- You can make pretty much any instrument that exists naturally and a number of instruments that don't.
- You can save and load instruments. (It's just an ordinary JSON file, so I guess you can also edit saved instruments with a text editor if you feel like it.)
- You can generate sounds on the fly with the "Generate and Play Sample" button, which just makes the noise (good for playing around and finding sounds).
- You can graph the frequency bands on a sound. (There is a similar feature in Audacity called "Plot Spectrum.")
- You can export your sounds as soundfont files (.sfz, which converts to .sf2)

## How-to

It's easier to just play around with it than to explain. Move the sliders and hit the Generate and Play Sample button. 

When you generate a soundfont, it also saves it to JSON automatically. When asked for a filename, put the filename without the .sfz extension. Good practice is to give each sound its own folder since it also generates the consitiuent .wav sound files for each note in the same directory, and they can make things cluttered.
