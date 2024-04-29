# Import packages
import streamlit as st
import pandas as pd
import numpy as np
from sympy import *

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

# Session states
ss = st.session_state
initial_variables = {'select_all':False, 'simulate_button':False}
for var,val in initial_variables:
    if var not in ss:
        ss[var] = val

def activate(var, val):
    ss.var = val

# Title
st.image('https://github.com/eltgnd/sdg-markov/blob/master/logo.png?raw=true',width=100)
st.title('SDG+Markov')
with st.container(border=True):
    st.write('Driving Sustainability Forward: Southeast Asia\'s Sustainable Development Goals Progression through Markovian Correlations')

# Results


st.divider()
st.caption('INTERACTIVE SIMULATION')

# User Input
select_all = st.toggle('Choose specific SDG/s')
activate('select_all', select_all)

if not ss.select_all:
    with st.form('simulation'):
        sdg = st.multiselect('Select an SDG',
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
            help='Disclaimer: SDG 17 is not included due to limited data.',
            max_selections=15)
        t = st.number_input('Timestep', 1,50,1,1, help='Timestep refers to number of years to predict from 2023.')

        s = st.form_submit_button("Simulate")
    if s:
        activate('simulate_button', s)
    
# Simulation
df = pd.read_csv('data.csv')

# Simulation variables
labels = {'Not Started':0, 'On Track':1, 'Needs Acceleration':2, 'Regresses':3}
n = len(labels)
current_year = 2023

def convert_sdg_states(sdg):
  lst = df[df.UNSDG == sdg]['State']
  new_lst = []

  for i in lst:
    new_lst.append(labels[i])
  return new_lst

# Create 1x4 initial state vector based on current/most recent data
def create_initial_state_vector(sdg,year=current_year):
  state = df[(df.UNSDG == sdg) & (df.Year == year)]['State']
  state = state.to_string().split('    ')[-1] # ugly solution, will probably fix
  labels = {'Not Started':0, 'On Track':1, 'Needs Acceleration':2, 'Regresses':3}
  v = [0]*n
  v[labels[state]] = 1
  return v

# Create 4x4 transition matrix based on transition probabilities
def create_transition_matrix(transitions):
  M = [[0]*n for i in range(n)]

  for (i,j) in zip(transitions, transitions[1:]):
    M[i][j] += 1
  for row in M:
    s = sum(row)
    if s > 0:
      row[:] = [f/s for f in row]

  return [list(i) for i in zip(*M)]

def markov_chain(sdg, N):
  isv = create_initial_state_vector(sdg)
  tm = create_transition_matrix(convert_sdg_states(sdg))

  current_state = isv
  states = {}

  for i in range(N):
    new_state = np.matmul(current_state, tm)
    states[i] = new_state
    current_state = new_state

  return states