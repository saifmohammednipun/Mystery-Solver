import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD

# Streamlit UI title
st.title("Mystery Solver: Bayesian Network")

# Define Bayesian Network Structure
model = BayesianNetwork([
    ('Chase', 'Decker'),  # Rule 2
    ('Chase', 'Mullaney'),  # Rule 3
    ('Heath', 'Mullaney'),  # Rule 4
    ('Chase', 'Heath'),  # Rule 5 (not both guilty)
    ('Heath', 'Decker')  # Rule 6
])

# Define CPDs based on the given constraints
cpd_chase = TabularCPD(variable='Chase', variable_card=2, values=[[0.5], [0.5]])
cpd_heath = TabularCPD(variable='Heath', variable_card=2, values=[[0.5], [0.5]])
cpd_mullaney = TabularCPD(variable='Mullaney', variable_card=2,
                           values=[[1, 1, 0, 0], [0, 0, 1, 1]],
                           evidence=['Chase', 'Heath'], evidence_card=[2, 2])
cpd_decker = TabularCPD(variable='Decker', variable_card=2,
                         values=[[1, 0, 1, 0], [0, 1, 0, 1]],
                         evidence=['Chase', 'Heath'], evidence_card=[2, 2])

# Add CPDs to the model
model.add_cpds(cpd_chase, cpd_heath, cpd_mullaney, cpd_decker)

# Verify model validity
assert model.check_model()

# Perform inference
inference = VariableElimination(model)
if st.button("Solve Mystery"):
    guilty_prob = inference.map_query(variables=['Chase', 'Decker', 'Heath', 'Mullaney'])
    guilty_suspects = [suspect for suspect, guilty in guilty_prob.items() if guilty == 1]
    st.write("### Most Likely Guilty Suspects")
    st.write(", ".join(guilty_suspects) if guilty_suspects else "No suspects found guilty.")

# Visualize Bayesian Network
st.write("### Bayesian Network Graph")
G = nx.DiGraph()
G.add_edges_from(model.edges())
plt.figure(figsize=(5, 3))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='black', node_size=3000, font_size=10)
st.pyplot(plt)
