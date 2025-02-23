import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import ParameterEstimator
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD

# Title of the application
st.title("Mystery Solver: Bayesian Network for Fictional Mysteries")

# Predefined Fictional Mystery Scenarios
scenarios = {
    "Stolen Artifact": {
        "description": "A priceless artifact was stolen from the museum last night.",
        "suspects": ["Alice", "Bob", "Charlie"],
    },
    "Missing Heirloom": {
        "description": "A family heirloom went missing during a dinner party.",
        "suspects": ["David", "Eve", "Frank"],
    },
    "Bank Robbery": {
        "description": "A bank was robbed, and the suspects fled in a getaway car.",
        "suspects": ["Grace", "Hank", "Ivy"],
    },
}

# Input Section
st.header("Input Section")

# 1. Choose a Fictional Mystery Scenario
selected_scenario = st.selectbox(
    "Choose a fictional mystery scenario:", list(scenarios.keys())
)

# Display the scenario description
st.write(f"**Scenario Description:** {scenarios[selected_scenario]['description']}")

# 2. Show All Suspects
suspects = scenarios[selected_scenario]["suspects"]
st.write(f"**Suspects:** {', '.join(suspects)}")

# Define the Bayesian Network structure
def create_bayesian_network():
    model = BayesianNetwork([
        ('Motive', 'Guilty'),
        ('Opportunity', 'Guilty'),
        ('Evidence', 'Guilty')
    ])

    # Define the Conditional Probability Tables (CPTs)
    cpd_motive = TabularCPD(variable='Motive', variable_card=2, values=[[0.7], [0.3]])
    cpd_opportunity = TabularCPD(variable='Opportunity', variable_card=2, values=[[0.6], [0.4]])
    cpd_evidence = TabularCPD(variable='Evidence', variable_card=2, values=[[0.8], [0.2]])
    cpd_guilty = TabularCPD(variable='Guilty', variable_card=2,
                             values=[[0.9, 0.7, 0.6, 0.3, 0.5, 0.2, 0.1, 0.05],
                                     [0.1, 0.3, 0.4, 0.7, 0.5, 0.8, 0.9, 0.95]],
                             evidence=['Motive', 'Opportunity', 'Evidence'],
                             evidence_card=[2, 2, 2])

    model.add_cpds(cpd_motive, cpd_opportunity, cpd_evidence, cpd_guilty)
    return model

# Streamlit UI
def main():
    model = create_bayesian_network()
    inference = VariableElimination(model)
    
    # User Inputs
    motive = st.selectbox("Does the suspect have a motive?", ["Unknown", "Yes", "No"])
    opportunity = st.selectbox("Did the suspect have the opportunity?", ["Unknown", "Yes", "No"])
    evidence = st.selectbox("Is there any direct evidence?", ["Unknown", "Yes", "No"])
    
    evidence_dict = {}
    if motive != "Unknown":
        evidence_dict['Motive'] = 1 if motive == "Yes" else 0
    if opportunity != "Unknown":
        evidence_dict['Opportunity'] = 1 if opportunity == "Yes" else 0
    if evidence != "Unknown":
        evidence_dict['Evidence'] = 1 if evidence == "Yes" else 0
    
    if st.button("Solve Mystery"):
        result = inference.query(variables=['Guilty'], evidence=evidence_dict)
        st.write("### Probability of Being Guilty")
        st.write(result)
    
    # Show Bayesian Network Graph
    st.write("### Bayesian Network Graph")
    G = nx.DiGraph()
    G.add_edges_from([('Motive', 'Guilty'), ('Opportunity', 'Guilty'), ('Evidence', 'Guilty')])
    plt.figure(figsize=(5, 3))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='black', node_size=3000, font_size=10)
    st.pyplot(plt)

if __name__ == "__main__":
    main()