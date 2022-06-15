import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
st.set_page_config(layout = 'wide',page_title='Predictive Analytics on Airport Machines')
import UPS
import chiller

with st.sidebar:
    machine = option_menu('Machine',['Chiller', 'UPS'],menu_icon='hdd-stack-fill',icons=['fan','plug-fill'])

col1,col2,col3 = st.columns([3,6,3])

Logo = Image.open('resoluteai.png')
client = Image.open('Airport.png')
col1.image(client)
col3.image(Logo)
col2.markdown("<h1 style = 'text-align:center'> Predictive Analytics on Machines<h1>",unsafe_allow_html=True)

if machine == 'Chiller':
    # st.write('#### You are viewing the chiller page')
    chiller.UI()

elif machine == 'UPS':
    UPS.UI()

    