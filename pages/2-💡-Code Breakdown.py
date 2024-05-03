import streamlit as st
import pandas as pd

# Page config
st.set_page_config(
    page_title="Code Breakdown",
    page_icon="💡",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Title
st.title('💡')
st.title('Code Breakdown')

# Data used
with st.expander('Data gathering'):

    st.write('States of the each SDG in Southeast Asia from 2017 to 2023')
    sdg_data = pd.read_csv('data.csv')
    st.dataframe(sdg_data, height=200, use_container_width=True)

    st.write('A correlation matrix of synergies and trade-offs of each SDG')
    corr = pd.read_csv('correlation_matrix.csv')
    st.dataframe(corr, height=200, use_container_width=True)

# Set-up
with st.expander('Set-up'):
    st.write('From the dataframe, we start setting-up the requirements for conducting a Markov chain simulation. The first function converts the transition labels into numbers for processing later on.')

    st.code("""
def convert_sdg_states(sdg):
  lst = df[df.UNSDG == sdg]['State']
  new_lst = []

  for i in lst:
    new_lst.append(labels[i])
  return new_lst
    """)

    st.write('Then, we create the necessary elements for our Markov chain simulation: the initial state vector and transition probabiltiy matrix for each SDG.')

    st.code("""
def create_initial_state_vector(sdg,year=current_year):
  state = df[(df.UNSDG == sdg) & (df.Year == year)]['State']
  state = state.to_string().split('    ')[-1] # ugly solution, will probably fix
  labels = {'Not Started':0, 'On Track':1, 'Needs Acceleration':2, 'Regresses':3}
  v = [0]*n
  v[labels[state]] = 1
  return v
    """)

    st.code("""
def create_transition_matrix(transitions):
  M = [[0]*n for i in range(n)]

  for (i,j) in zip(transitions, transitions[1:]):
    M[i][j] += 1
  for row in M:
    s = sum(row)
    if s > 0:
      row[:] = [f/s for f in row]

  return [list(i) for i in zip(*M)]""")

with st.expander('Simulation'):
    st.write('Once set-up is complete, we can now conduct a Markov chain simulation. Given an SDG number and a timestep, the snippet below uses the previous functions to generate the state probabilities for an SDG for N timesteps.')
    st.code("""
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
    """
    )

    st.write('Then, the following snippets were used to export the simulation results as a CSV file for analysis.')
    st.code("""
    prediction_matrices = {}
num_timesteps = 10

for i in range(1,18):
  prediction_matrices[i] = markov_chain(i,num_timesteps)
  """)
    st.code("""
def get_sdg_list():
  lst = []
  for i in range(1,18):
    for j in range(num_timesteps):
      lst.append(i)
  return lst

def get_result(ind):
  lst = []
  for sdg in list(prediction_matrices.values()):
    for result in list(sdg.values()):
      lst.append(result[ind])
  return lst

cols = [f'Timestep {i+1}' for i in range(num_timesteps)]

results = pd.DataFrame(
    data={
        'SDG': get_sdg_list(),
        'Timestep': [i for i in range(num_timesteps)] * 17,
        'Not Started': get_result(0),
        'On Track': get_result(1),
        'Needs Acceleration': get_result(2),
        'Regresses': get_result(3)
      }
    )
    """)

    st.write('This produces the table used for the previous pages in this app.')

with st.expander('Correlation matrix analysis'):
    st.write('After conducting the simulation, the referenced correlation matrix was used to analyze the results. The following snippet shows the process of "transcribing" the correlation matrix from a txt file.')
    st.code("""
# Import text file containing the values for the correlation matrix.
# Text was extracted using https://www.imagetotext.io/

txt_file = open('correlation-matrix.txt', 'r')
corr_mat_str = txt_file.read()

# Cleaning
corr_mat_str = corr_mat_str.replace('\n', ' ')
l = corr_mat_str.split(' ')

h = []
for i in l:
  if len(i) != 0 and i[0] in ['1', '0', '-']:
    h.append(float(i))
print(len(h) == 16**2) # Check if the formed list contains same number of elements

corr_mat = []
for i in range(16):
  corr_mat.append(h[i*16:(i+1)*16])
    """)

    st.write("Then, functions were created to conduct analysis. The following snippet collects the 'contribution' of every other SDG to a specific SDG based on its state distribution vector.")
    st.code("""
def sdg_corr(sdg, m, lst=corr_mat):
  # Get the most likely state
  ind = m.index(max(m))
  row = lst[sdg-1]
  sign = 'negative' if ind == 3 else 'positive' # ind == 3 implies the "Regressed" state
  val = {}

  if sign == 'positive':
    # For all SDGs, store the correlation if it is positive
    # Note that range(16) is a consequence of the correlation matrix only available until SDG 16 and not SDG 17.
    for i in range(16):
      if row[i] > 0:
        val[i+1] = row[i]
  else:
    # For all SDGs, store the correlation if it is negative
    for i in range(16):
      if row[i] < 0:
        val[i+1] = row[i]

  total = sum(val.values())

  p = {}
  for sdg,i in val.items():
    p[sdg] = i/total

  return p, sign
    """)

    st.write("Then, the following function just uses the previous function to sum up the contribution percentage of each SDG for analysis.")
    st.code("""

def sdg_sum(prediction_matrices, chosen_timestep):
  # Create dictionaries
  sdg_total_positive, sdg_total_negative = {}, {}
  sdg_positive, sdg_negative = {}, {}
  for i in range(16):
    sdg_total_positive[i+1] = 0
    sdg_total_negative[i+1] = 0
    sdg_positive[i+1] = {key:0 for key in [k+1 for k in range(16)]}
    sdg_negative[i+1] = {key:0 for key in [k+1 for k in range(16)]}

  for s in range(16):
    sdg, markov_chain = s+1, prediction_matrices[s+1]
    m = markov_chain[chosen_timestep]
    p, sign = sdg_corr(sdg,m.tolist())

    if sign == 'positive':
      for i,j in p.items():
        sdg_total_positive[i] += j
        sdg_positive[i][sdg] += j
    else:
      for i,j in p.items():
        sdg_total_negative[i] += j
        sdg_negative[i][sdg] += j

  return sdg_total_positive, sdg_total_negative, sdg_positive, sdg_negative
  """)

