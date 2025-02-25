import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD

def create_bayesian_network():
    model = BayesianNetwork([
        ('Motive', 'Guilty'),
        ('Opportunity', 'Guilty'),
        ('Evidence', 'Guilty'),
        ('Alibi', 'Guilty'),
        ('Witness', 'Guilty')
    ])

    # Define Conditional Probability Tables (CPTs)
    cpd_motive = TabularCPD(variable='Motive', variable_card=2, values=[[0.6], [0.4]])
    cpd_opportunity = TabularCPD(variable='Opportunity', variable_card=2, values=[[0.7], [0.3]])
    cpd_evidence = TabularCPD(variable='Evidence', variable_card=2, values=[[0.8], [0.2]])
    cpd_alibi = TabularCPD(variable='Alibi', variable_card=2, values=[[0.5], [0.5]])
    cpd_witness = TabularCPD(variable='Witness', variable_card=2, values=[[0.6], [0.4]])
   
    # Ensure correct shape for Guilty (2 x 32)
    cpd_guilty = TabularCPD(
        variable='Guilty', variable_card=2,
        values=[
            [0.9, 0.8, 0.75, 0.6, 0.7, 0.55, 0.5, 0.4,
             0.8, 0.7, 0.65, 0.5, 0.6, 0.45, 0.4, 0.3,
             0.7, 0.6, 0.55, 0.4, 0.5, 0.35, 0.3, 0.2,
             0.6, 0.5, 0.45, 0.3, 0.4, 0.25, 0.2, 0.1],
            [0.1, 0.2, 0.25, 0.4, 0.3, 0.45, 0.5, 0.6,
             0.2, 0.3, 0.35, 0.5, 0.4, 0.55, 0.6, 0.7,
             0.3, 0.4, 0.45, 0.6, 0.5, 0.65, 0.7, 0.8,
             0.4, 0.5, 0.55, 0.7, 0.6, 0.75, 0.8, 0.9]
        ],
        evidence=['Motive', 'Opportunity', 'Evidence', 'Alibi', 'Witness'],
        evidence_card=[2, 2, 2, 2, 2]
    )

    model.add_cpds(cpd_motive, cpd_opportunity, cpd_evidence, cpd_alibi, cpd_witness, cpd_guilty)
    model.check_model()
    return model

def main():
    st.title("Mystery Solver: Bayesian Network")
    model = create_bayesian_network()
    inference = VariableElimination(model)

    # User Inputs
    motive = st.selectbox("Does the suspect have a motive?", ["Unknown", "Yes", "No"])
    opportunity = st.selectbox("Did the suspect have the opportunity?", ["Unknown", "Yes", "No"])
    evidence = st.selectbox("Is there any direct evidence?", ["Unknown", "Yes", "No"])
    alibi = st.selectbox("Does the suspect have an alibi?", ["Unknown", "Yes", "No"])
    witness = st.selectbox("Did a witness testify against the suspect?", ["Unknown", "Yes", "No"])

    evidence_dict = {}
    if motive != "Unknown":
        evidence_dict['Motive'] = 1 if motive == "Yes" else 0
    if opportunity != "Unknown":
        evidence_dict['Opportunity'] = 1 if opportunity == "Yes" else 0
    if evidence != "Unknown":
        evidence_dict['Evidence'] = 1 if evidence == "Yes" else 0
    if alibi != "Unknown":
        evidence_dict['Alibi'] = 1 if alibi == "Yes" else 0
    if witness != "Unknown":
        evidence_dict['Witness'] = 1 if witness == "Yes" else 0

    if st.button("Solve Mystery"):
        result = inference.query(variables=['Guilty'], evidence=evidence_dict)
        st.write("### Probability of Being Guilty")
        st.write(result)

    # Bayesian Network Graph
    st.write("### Bayesian Network Graph")
    G = nx.DiGraph()
    G.add_edges_from([
        ('Motive', 'Guilty'), ('Opportunity', 'Guilty'), ('Evidence', 'Guilty'), ('Alibi', 'Guilty'), ('Witness', 'Guilty')
    ])
    plt.figure(figsize=(6, 4))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='black', node_size=3000, font_size=10)
    st.pyplot(plt)

if __name__ == "__main__":
    main()
