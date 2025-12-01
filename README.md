# Lavalamp Webcam Entropy Experiment

This project generates entropy bytes from a webcam feed and writes them into a named pipe.  
Other programs, such as U-ai, can read these bytes to introduce external randomness.

## Purpose

The goal is to create a simple real-world entropy source using webcam pixel data.  
This is meant for experimentation and not for cryptographic use.

## How It Works

1. The Python script opens the webcam.
2. Each frame is sampled by averaging pixel intensity.
3. One entropy byte is derived.
4. The byte is written into:

       /tmp/unhidra_entropy.pipe

5. Any external program can read from this pipe continuously.

## Running

Create the pipe:

    rm -f /tmp/unhidra_entropy.pipe
    mkfifo /tmp/unhidra_entropy.pipe

Start the entropy streamer:

    python3 src/entropy_pipe.py

It will run continuously and write one byte at a time.

## Requirements

- Python 3
- OpenCV (opencv-python)
- NumPy
- A working webcam

## Project Structure

- src/entropy_pipe.py — main entropy generator.
- src/modules/camera_capture.py — webcam utilities.
- src/modules/entropy_pool.py — simple entropy processing logic.

## Notes

This entropy system is experimental and not designed for security-critical randomness.
