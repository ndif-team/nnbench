"""The tests run without GPUs: hide them before anything initializes CUDA, so a test never
touches a GPU that a timed benchmark run holds."""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
