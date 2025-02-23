import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD

def build_bayesian_network():
    """Constructs the Bayesian Network for the Mystery Solver."""
    model = BayesianNetwork([
        ('Motive', 'Guilty'),
        ('Opportunity', 'Guilty'),
        ('Alibi', 'Guilty'),
        ('WitnessTestimony', 'Guilty'),
        ('Guilty', 'Arrest')
    ])
    
    # Conditional Probability Tables (CPTs)
    cpd_motive = TabularCPD(variable='Motive', variable_card=2, values=[[0.6], [0.4]])
    cpd_opportunity = TabularCPD(variable='Opportunity', variable_card=2, values=[[0.7], [0.3]])
    cpd_alibi = TabularCPD(variable='Alibi', variable_card=2, values=[[0.8], [0.2]])
    cpd_witness = TabularCPD(variable='WitnessTestimony', variable_card=2, values=[[0.9], [0.1]])
    
    cpd_guilty = TabularCPD(
        variable='Guilty', variable_card=2,
        values=[
            [0.99, 0.8, 0.85, 0.7, 0.75, 0.6, 0.65, 0.5, 0.55, 0.4, 0.45, 0.3, 0.35, 0.2, 0.25, 0.1],
            [0.01, 0.2, 0.15, 0.3, 0.25, 0.4, 0.35, 0.5, 0.45, 0.6, 0.55, 0.7, 0.65, 0.8, 0.75, 0.9]
        ],
        evidence=['Motive', 'Opportunity', 'Alibi', 'WitnessTestimony'],
        evidence_card=[2, 2, 2, 2]
    )
    
    cpd_arrest = TabularCPD(
        variable='Arrest', variable_card=2,
        values=[[0.95, 0.3], [0.05, 0.7]],
        evidence=['Guilty'], evidence_card=[2]
    )
    
    # Add CPDs to the model
    model.add_cpds(cpd_motive, cpd_opportunity, cpd_alibi, cpd_witness, cpd_guilty, cpd_arrest)
    
    # Verify the model
    assert model.check_model()
    return model

def infer_guilt(model, evidence):
    """Performs inference to determine the probability of guilt given the evidence."""
    inference = VariableElimination(model)
    result = inference.query(variables=['Guilty'], evidence=evidence)
    return result

def draw_factor_graph(model):
    """Draws the factor graph of the Bayesian Network."""
    G = nx.DiGraph()
    G.add_edges_from(model.edges())
    plt.figure(figsize=(6, 4))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=3000, font_size=10)
    st.pyplot(plt)

st.title("Mystery Solver")
st.write("Enter evidence to determine the likelihood of guilt.")

bayesian_model = build_bayesian_network()

evidence = {}
evidence['Motive'] = st.selectbox("Motive", [0, 1])
evidence['Opportunity'] = st.selectbox("Opportunity", [0, 1])
evidence['Alibi'] = st.selectbox("Alibi", [0, 1])
evidence['WitnessTestimony'] = st.selectbox("Witness Testimony", [0, 1])

if st.button("Solve Mystery"):
    result = infer_guilt(bayesian_model, evidence)
    st.write("### Inference Result:")
    st.write(result)
    st.write("### Factor Graph:")
    draw_factor_graph(bayesian_model)
