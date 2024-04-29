# Import packages
import streamlit as st

# Page config
st.set_page_config(
    page_title="SDG+Markov",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

# Title
# st.image(,width=100)
st.title('SDG+Markov')
with st.container(border=True):
    st.write('Driving Sustainability Forward: Southeast Asia’s Sustainable Development Goals Progression through Markovian Correlations')
