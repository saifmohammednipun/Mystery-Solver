import streamlit as st
import pandas as pd
import numpy as np
# Corrected import for newer pgmpy versions
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import traceback # For better error printing if needed
import graphviz # <-- *** ADDED IMPORT ***

# --- Bayesian Network Definition ---
# (build_bayesian_network function remains the same as before)
@st.cache_data # Cache the BN creation for efficiency
def build_bayesian_network():
    """
    Defines and builds the Bayesian Network structure and CPTs for the mystery.
    """
    # 1. Define Variables and Structure (Nodes and Edges)
    model = DiscreteBayesianNetwork([
        ('Motive_A', 'Guilty_A'), ('Opportunity_A', 'Guilty_A'), # Alice's factors
        ('Motive_B', 'Guilty_B'), ('Opportunity_B', 'Guilty_B'), # Bob's factors
        ('Guilty_A', 'Crumbs_Near_A'),                          # Evidence linked to Alice
        ('Guilty_B', 'Witness_Saw_B')                           # Evidence linked to Bob
    ])

    # 2. Define Conditional Probability Distributions (CPDs)
    # ... (CPD definitions are exactly the same as before) ...
    # --- Priors (Nodes with no parents) ---
    cpd_m_a = TabularCPD(variable='Motive_A', variable_card=2, values=[[0.4], [0.6]])
    cpd_o_a = TabularCPD(variable='Opportunity_A', variable_card=2, values=[[0.3], [0.7]])
    cpd_m_b = TabularCPD(variable='Motive_B', variable_card=2, values=[[0.5], [0.5]])
    cpd_o_b = TabularCPD(variable='Opportunity_B', variable_card=2, values=[[0.4], [0.6]])

    # --- Conditionals ---
    # P(Guilty_A | Motive_A, Opportunity_A)
    cpd_g_a = TabularCPD(variable='Guilty_A', variable_card=2,
                         values=[[0.99, 0.8, 0.7, 0.2], [0.01, 0.2, 0.3, 0.8]],
                         evidence=['Motive_A', 'Opportunity_A'], evidence_card=[2, 2])

    # P(Guilty_B | Motive_B, Opportunity_B)
    cpd_g_b = TabularCPD(variable='Guilty_B', variable_card=2,
                         values=[[0.98, 0.75, 0.7, 0.1], [0.02, 0.25, 0.3, 0.9]],
                         evidence=['Motive_B', 'Opportunity_B'], evidence_card=[2, 2])

    # P(Crumbs_Near_A | Guilty_A)
    cpd_c_a = TabularCPD(variable='Crumbs_Near_A', variable_card=2,
                         values=[[0.9, 0.3], [0.1, 0.7]],
                         evidence=['Guilty_A'], evidence_card=[2])

    # P(Witness_Saw_B | Guilty_B)
    cpd_w_b = TabularCPD(variable='Witness_Saw_B', variable_card=2,
                         values=[[0.95, 0.2], [0.05, 0.8]],
                         evidence=['Guilty_B'], evidence_card=[2])

    # 3. Add CPDs to the model
    model.add_cpds(cpd_m_a, cpd_o_a, cpd_m_b, cpd_o_b, cpd_g_a, cpd_g_b, cpd_c_a, cpd_w_b)

    # 4. Check model validity (important!)
    try:
        if not model.check_model():
            raise ValueError("Bayesian Network structure or CPDs are invalid (check_model failed).")
        print("Bayesian Network Built Successfully!")
    except Exception as e:
        print(f"Error during model check: {e}")
        raise ValueError(f"Bayesian Network validation failed: {e}")

    return model

# --- Inference Function ---
# (perform_inference function remains the same as before)
def perform_inference(model, evidence_dict):
    """
    Performs probabilistic inference on the Bayesian Network.
    (Code is exactly the same as the previous version)
    """
    if not evidence_dict:
        print("Performing inference with no evidence (calculating priors/marginals).")
    else:
        valid_evidence = {k: int(v) for k, v in evidence_dict.items() if v is not None and k in model.nodes()}
        print(f"Performing inference with evidence: {valid_evidence}")
        if len(valid_evidence) != len(evidence_dict):
            print("Warning: Some provided evidence keys were not found in the model nodes or were None.")

    try:
        inference_engine = VariableElimination(model)
    except Exception as e:
        st.error(f"Failed to initialize inference engine: {e}")
        print(f"Failed to initialize inference engine: {e}\n{traceback.format_exc()}")
        return None

    query_variables = ['Guilty_A', 'Guilty_B']
    valid_query_variables = [q for q in query_variables if q in model.nodes()]
    if not valid_query_variables:
        st.error("None of the query variables are in the model.")
        print("Error: None of the query variables found in the model.")
        return None

    try:
        results = inference_engine.query(variables=valid_query_variables, evidence=valid_evidence)
        print("Inference successful.")
        results_dict = {}
        if isinstance(results, dict):
            results_dict = results
        elif valid_query_variables:
             results_dict = {valid_query_variables[0]: results}
        return results_dict

    except Exception as e:
        st.error(f"Inference failed: {e}")
        print(f"Inference failed: {e}\n{traceback.format_exc()}")
        if hasattr(e, 'factors'):
             print("Factors involved:", e.factors)
        return None

# --- Helper function to create Graphviz object ---
def create_graphviz_object(model):
    """Creates a graphviz.Digraph object from the pgmpy model."""
    dot = graphviz.Digraph(comment='Bayesian Network')
    dot.attr(rankdir='TB') # Top-to-Bottom layout

    # Add nodes
    for node in model.nodes():
        dot.node(node, node) # Use node name as both ID and label

    # Add edges
    for edge in model.edges():
        dot.edge(edge[0], edge[1])

    return dot


# --- Streamlit User Interface ---

st.set_page_config(page_title="Bayesian Mystery Solver", layout="wide")

st.title("🕵️‍♂️ Bayesian Mystery Solver")
st.markdown("""
Welcome! This application uses a **Discrete Bayesian Network** to solve a fictional mystery: **"Who Stole the Cookie?"**

Based on the concepts taught in the AI course (Volume 4):
*   **Bayesian Networks:** The relationships between suspects, motives, opportunities, and evidence are modeled as a Directed Acyclic Graph (DAG).
*   **Probability:** We use Conditional Probability Tables (CPTs) to define the likelihood of events (e.g., P(Crumbs Found | Guilty)).
*   **Probabilistic Inference:** Enter the evidence you've gathered (observations). The system will calculate the updated probability of each suspect being guilty, $P(\text{Guilty} | \text{Evidence})$, using conditioning and marginalization.

**Scenario:** Alice and Bob are suspects. Let's find the likely culprit!
""")

# Build the Bayesian Network
try:
    bn_model = build_bayesian_network()
    st.success("Bayesian Network model loaded successfully!")

    # --- UI Elements ---
    evidence_options = {
        'Motive_A': "Alice had Motive?",
        'Opportunity_A': "Alice had Opportunity?",
        'Motive_B': "Bob had Motive?",
        'Opportunity_B': "Bob had Opportunity?",
        'Crumbs_Near_A': "Cookie Crumbs found near Alice?",
        'Witness_Saw_B': "Witness saw Bob near the scene?"
    }
    state_options = ["Unknown", "Yes", "No"]
    evidence_values = {}

    st.sidebar.header("Enter Evidence:")
    for var, label in evidence_options.items():
        if var in bn_model.nodes():
            user_input = st.sidebar.selectbox(label, options=state_options, index=0, key=var)
            if user_input == "Yes":
                evidence_values[var] = 1
            elif user_input == "No":
                evidence_values[var] = 0
        else:
             st.sidebar.warning(f"Variable '{var}' defined in UI but not in BN model.")

    evidence_for_inference = {k: v for k, v in evidence_values.items() if v is not None}

    st.header("Evidence Gathered:")
    if not evidence_for_inference:
        st.info("No evidence entered yet. Showing prior probabilities (or marginals).")
    else:
        display_evidence = {k: ("Yes" if v==1 else "No") for k,v in evidence_for_inference.items()}
        st.json(display_evidence)

    # Perform Inference when button is clicked
    if st.button("Solve Mystery / Update Probabilities"):
        st.header("Inference Results:")
        with st.spinner("Calculating probabilities..."):
            posterior_probs = perform_inference(bn_model, evidence_for_inference)

            if posterior_probs:
                prob_a_guilty = -1.0
                prob_b_guilty = -1.0
                prob_g_a_factor = posterior_probs.get('Guilty_A')
                prob_g_b_factor = posterior_probs.get('Guilty_B')

                # Extract probability of state 1 (Yes)
                # (Extraction logic remains the same as before)
                if prob_g_a_factor is not None:
                    try:
                        # Using get_value ensures correctness even if factor dimensions change
                        prob_a_guilty = prob_g_a_factor.get_value(**{'Guilty_A': 1})
                    except Exception as e:
                        st.warning(f"Could not extract P(Guilty_A=Yes): {e}")
                        print(f"Error extracting P(Guilty_A=Yes): {e}\nFactor G_A:\n{prob_g_a_factor}")
                if prob_g_b_factor is not None:
                     try:
                        prob_b_guilty = prob_g_b_factor.get_value(**{'Guilty_B': 1})
                     except Exception as e:
                         st.warning(f"Could not extract P(Guilty_B=Yes): {e}")
                         print(f"Error extracting P(Guilty_B=Yes): {e}\nFactor G_B:\n{prob_g_b_factor}")

                # Display Metrics
                if prob_a_guilty >= 0:
                     st.metric(label="Probability Alice is Guilty (P(Guilty_A=Yes | Evidence))", value=f"{prob_a_guilty:.2%}")
                else:
                     st.warning("Could not calculate probability for Alice's guilt.")
                if prob_b_guilty >= 0:
                     st.metric(label="Probability Bob is Guilty (P(Guilty_B=Yes | Evidence))", value=f"{prob_b_guilty:.2%}")
                else:
                     st.warning("Could not calculate probability for Bob's guilt.")

                # Display the full probability distributions (optional)
                with st.expander("Show Full Probability Distributions (Factors)"):
                    if prob_g_a_factor is not None: st.text(f"P(Guilty_A | Evidence):\n{prob_g_a_factor}")
                    if prob_g_b_factor is not None: st.text(f"P(Guilty_B | Evidence):\n{prob_g_b_factor}")
            else:
                st.error("Inference could not be completed. Check logs or CPTs.")

    # --- Display Model Info ---
    st.markdown("---")
    st.markdown("Note: Probabilities are based on the defined network structure and CPTs. "
                "Adjusting the CPTs or network structure would change the results.")

    # --- *** NEW VISUALIZATION SECTION *** ---
    st.subheader("Bayesian Network Visualization")
    st.info("Requires Graphviz system tools to be installed (see code comments/readme).")
    try:
        graph_obj = create_graphviz_object(bn_model)
        st.graphviz_chart(graph_obj)
    except ImportError:
        st.warning("Python `graphviz` package not found. Please install it (`pip install graphviz`).")
    except graphviz.backend.execute.ExecutableNotFound:
        st.error("Graphviz executable not found. Please install Graphviz system software and ensure it's in your PATH.")
        st.markdown("Download from: [https://graphviz.org/download/](https://graphviz.org/download/)")
    except Exception as e:
         st.error(f"Could not render graph: {e}")
         print(f"Error rendering graph: {e}\n{traceback.format_exc()}")
    # --- End of New Visualization Section ---

    with st.expander("Show Bayesian Network Structure (Edges List)"):
         st.write(bn_model.edges())
    with st.expander("Show Conditional Probability Distributions (CPDs)"):
         for cpd in bn_model.get_cpds():
             st.text(str(cpd))

except Exception as e:
    st.error(f"An error occurred during application setup: {e}")
    st.error("This might be the Bayesian Network definition or a library conflict.")
    print(f"Critical error during setup: {e}\n{traceback.format_exc()}")