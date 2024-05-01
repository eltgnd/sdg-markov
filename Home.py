# Import packages
import streamlit as st
import pandas as pd
import numpy as np
from simulation import *

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

# Simulation variables
labels = {'Not Started':0, 'On Track':1, 'Needs Acceleration':2, 'Regresses':3}
n = len(labels)
current_year = 2023

# Helper functions
def list_to_latex_matrix(lst):
    latex_matrix = "\\begin{bmatrix}"
    for row in lst:
        if isinstance(row, list):  # check if row is a list
            for num in row:
                latex_matrix += str(num) + " & "
            latex_matrix = latex_matrix[:-2]  # remove the last "&"
        else:  # if row is not a list, it's an integer
            latex_matrix += str(row) + " & "
            latex_matrix = latex_matrix[:-2]  # remove the last "&"
        latex_matrix += "\\\\"
    latex_matrix += "\\end{bmatrix}"
    return latex_matrix

# Session states
ss = st.session_state
initial_variables = {'simulate_button':False}
for var,val in initial_variables.items():
    if var not in ss:
        ss[var] = val

def activate(var, val):
    ss[var] = val

# Title
st.image('https://github.com/eltgnd/sdg-markov/blob/master/logo.png?raw=true',width=100)
st.title('SDG+Markov')
with st.container(border=True):
    st.write('Driving Sustainability Forward: Southeast Asia\'s Sustainable Development Goals Progression through Markovian Correlations')

# Results
st.divider()
st.caption('INTERACTIVE SIMULATION')

# User Input
# select_all = st.toggle('Choose specific SDG/s')

with st.form('simulation'):
    sdg = st.selectbox('Select an SDG',
        """SDG 1: No poverty
        SDG 2: Zero hunger
        SDG 3: Good health and well-being
        SDG 4: Quality education
        SDG 5: Gender equality
        SDG 6: Clean water and sanitation
        SDG 7: Affordable and clean energy
        SDG 8: Decent work and economic growth
        SDG 9: Industry, innovation, and infrastructure
        SDG 10: Reduced inequalities
        SDG 11: Sustainable cities and communities
        SDG 12: Responsible consumption and production
        SDG 13: Climate action
        SDG 14: Life below water
        SDG 15: Life on land
        SDG 16: Peace, justice, and strong institutions""".split('\n'),
        help='Disclaimer: SDG 17 is not included due to limited data.',)
    t = st.number_input('Timestep', 1,50,1,1, help='Timestep refers to number of years to predict from 2023.')

    s = st.form_submit_button("Simulate")
if s:
    activate('simulate_button', s)
    

# Simulation
df = pd.read_csv('data.csv')

# Set-up
with st.container(border=True):
    sdg_number = int(sdg.split()[1][:-1])
    st.caption(f'SET-UP FOR SDG {sdg_number}')

    initial_mat = create_initial_state_vector(df, sdg_number, current_year)
    trans_mat = create_transition_matrix(convert_sdg_states(df, sdg_number))
    
    col1, col2, col3 = st.columns([0.5,0.7,0.2])
    with col1:
        st.write(f"""Initial state vector $=$ ${list_to_latex_matrix(initial_mat)}$""")
    with col2:
        st.write(f"""Transition matrix $=$ ${list_to_latex_matrix(trans_mat)}$""")
    st.write('\n')

# Markov chain result
with st.container(border=True):
    st.caption('MARKOV CHAIN RESULT')

    result = markov_chain(df, sdg_number, t)
    st.dataframe(result, use_container_width=True)