import streamlit as st
import PIL
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
def app():
    st.title("Welcome to Landmark Recognition Web Application")
    img = PIL.Image.open("home.jpg")
    img = img.resize((1000,500))
    st.image(img)
