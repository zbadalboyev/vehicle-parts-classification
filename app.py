import streamlit as st
from fastai.vision.all import *
import platform
import pathlib
import pandas as pd
import plotly.express as px

plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath

# title
st.title('Avtoulov qismlarini tasniflovchi model')

#rasmni joylash
file = st.file_uploader('Rasm yuklash', type=['png', 'jpeg', 'jpg', 'gif', 'svg'])
if (file):
    img = PILImage.create(file)
    st.image(img)
    model = load_learner('./parts_model.pkl')
    
    pred, pred_id, probs = model.predict(img)
    st.success(f"Bashorat: {pred}")
    st.info(f"Ehtimollik: {probs[pred_id]*100:.1f}%")

    # plotting
    fig = px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)

