import numpy as np

# Simulation variables
labels = {'Not Started':0, 'On Track':1, 'Needs Acceleration':2, 'Regresses':3}
n = len(labels)
current_year = 2023

def convert_sdg_states(df,sdg):
  lst = df[df.UNSDG == sdg]['State']
  new_lst = []

  for i in lst:
    new_lst.append(labels[i])
  return new_lst

# Create 1x4 initial state vector based on current/most recent data
def create_initial_state_vector(df, sdg,year=current_year):
  state = df[(df.UNSDG == sdg) & (df.Year == year)]['State']
  state = state.to_string().split('    ')[-1]
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

def markov_chain(df, sdg, N):
  isv = create_initial_state_vector(df, sdg)
  tm = create_transition_matrix(convert_sdg_states(df, sdg))

  current_state = isv
  states = {}

  for i in range(N):
    new_state = np.matmul(current_state, tm)
    states[i] = new_state
    current_state = new_state

  return states