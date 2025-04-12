import streamlit as st
import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import traceback # For better error printing
import graphviz # For visualization

# --- Constants for State Mapping (Improves Readability) ---
# Using integers 0, 1, 2, 3 etc. internally for pgmpy
CULPRIT_MAP = {'None': 0, 'Scholar (A)': 1, 'Butler (B)': 2, 'Cat Burglar (C)': 3}
FINGERPRINT_MAP = {'None': 0, 'Scholar (A)': 1, 'Butler (B)': 2, 'Cat Burglar (C)': 3}
BINARY_MAP = {'No': 0, 'Yes': 1} # For Yes/No variables

# --- Bayesian Network Definition ---

@st.cache_data # Cache the BN creation for efficiency
def build_manuscript_bayesian_network():
    """
    Defines and builds the Bayesian Network for the Missing Manuscript case.
    Uses a single 'Culprit' node for mutual exclusivity.
    """
    # 1. Define Variables and Structure (Nodes and Edges)
    # 'Culprit' has states: 0:None, 1:Scholar(A), 2:Butler(B), 3:Cat Burglar(C)
    # 'Fingerprints' has states: 0:None, 1:A, 2:B, 3:C
    # Other evidence nodes are binary (0:No, 1:Yes)

    model = DiscreteBayesianNetwork([
        # Culprit is a central node influencing evidence
        ('Culprit', 'Forced_Entry'),
        ('Culprit', 'Alibi_A'),      # Represents if A has a valid alibi
        ('Culprit', 'Alibi_B'),      # Represents if B has a valid alibi
        ('Culprit', 'Alibi_C'),      # Represents if C has a valid alibi
        ('Culprit', 'Fingerprints')
        # Independent root nodes representing general conditions or simple evidence
        # ('Motive_A'), ('Motive_B'), ('Security_Footage') # Optional: Could add these as roots if needed
    ])

    # 2. Define Conditional Probability Distributions (CPDs)

    # --- Prior Probability for the Culprit ---
    # P(Culprit) - Initial suspicion before any specific evidence
    # P(None)=0.05, P(A)=0.3, P(B)=0.3, P(C)=0.35 (sum=1) - Slightly higher suspicion for Burglar
    cpd_culprit = TabularCPD(variable='Culprit', variable_card=4,
                             values=[[0.05], [0.30], [0.30], [0.35]])

    # --- Evidence Nodes Conditioned on the Culprit ---

    # P(Forced_Entry | Culprit) - Binary (0:No, 1:Yes)
    # Burglar (C) most likely to force entry. Butler (B) least likely (has keys). Scholar (A) maybe.
    cpd_fe = TabularCPD(variable='Forced_Entry', variable_card=2,
                        values=[[0.99, 0.95, 0.98, 0.20],  # P(FE=No | Culprit)
                                [0.01, 0.05, 0.02, 0.80]], # P(FE=Yes | Culprit)
                        evidence=['Culprit'], evidence_card=[4])

    # P(Alibi_A | Culprit) - Binary (0:No Alibi, 1:Has Alibi)
    # If A is guilty (Culprit=1), less likely to have Alibi (P(Alibi_A=1) is low)
    cpd_alibi_a = TabularCPD(variable='Alibi_A', variable_card=2,
                             values=[[0.1, 0.8, 0.1, 0.1],  # P(Alibi_A=No | Culprit) -> High if A is culprit
                                     [0.9, 0.2, 0.9, 0.9]], # P(Alibi_A=Yes | Culprit) -> Low if A is culprit
                             evidence=['Culprit'], evidence_card=[4])

    # P(Alibi_B | Culprit) - Binary (0:No Alibi, 1:Has Alibi)
    # If B is guilty (Culprit=2), less likely to have Alibi
    cpd_alibi_b = TabularCPD(variable='Alibi_B', variable_card=2,
                             values=[[0.1, 0.1, 0.7, 0.1],  # P(Alibi_B=No | Culprit)
                                     [0.9, 0.9, 0.3, 0.9]], # P(Alibi_B=Yes | Culprit)
                             evidence=['Culprit'], evidence_card=[4])

    # P(Alibi_C | Culprit) - Binary (0:No Alibi, 1:Has Alibi)
    # If C is guilty (Culprit=3), very unlikely to have Alibi. Also generally less likely.
    cpd_alibi_c = TabularCPD(variable='Alibi_C', variable_card=2,
                             values=[[0.4, 0.4, 0.4, 0.9],  # P(Alibi_C=No | Culprit) - Higher overall, very high if C guilty
                                     [0.6, 0.6, 0.6, 0.1]], # P(Alibi_C=Yes | Culprit)
                             evidence=['Culprit'], evidence_card=[4])

    # P(Fingerprints | Culprit) - States: 0:None, 1:A, 2:B, 3:C
    # Defines the probability of finding specific prints given who the culprit is.
    cpd_fp = TabularCPD(variable='Fingerprints', variable_card=4,
                        values=[
                            # Probabilities P(FP=state | Culprit=state)
                            # FP=None | Culprit=None, A, B, C
                            [0.99, 0.60, 0.70, 0.50],
                            # FP=A    | Culprit=None, A, B, C
                            [0.003, 0.40, 0.00, 0.00],
                            # FP=B    | Culprit=None, A, B, C
                            [0.003, 0.00, 0.30, 0.00],
                            # FP=C    | Culprit=None, A, B, C
                            [0.004, 0.00, 0.00, 0.50]
                         ],
                        evidence=['Culprit'], evidence_card=[4])

    # --- Optional Independent Evidence Nodes (Can be added if desired) ---
    # cpd_motive_a = TabularCPD(variable='Motive_A', variable_card=2, values=[[0.3],[0.7]]) # P(Motive_A=No), P(Motive_A=Yes)=0.7
    # cpd_motive_b = TabularCPD(variable='Motive_B', variable_card=2, values=[[0.9],[0.1]]) # P(Motive_B=No), P(Motive_B=Yes)=0.1
    # cpd_sec_footage = TabularCPD(variable='Security_Footage', variable_card=2, values=[[0.4],[0.6]]) # P(SF=No), P(SF=Yes)=0.6

    # 3. Add CPDs to the model
    model.add_cpds(cpd_culprit, cpd_fe, cpd_alibi_a, cpd_alibi_b, cpd_alibi_c, cpd_fp)
                    # Optional: cpd_motive_a, cpd_motive_b, cpd_sec_footage)

    # 4. Check model validity
    try:
        if not model.check_model():
            raise ValueError("Bayesian Network structure or CPDs are invalid.")
        print("Manuscript Bayesian Network Built Successfully!")
    except Exception as e:
        print(f"Error during model check: {e}")
        raise ValueError(f"Bayesian Network validation failed: {e}")

    return model

# --- Inference Function (Mostly Reusable) ---
def perform_manuscript_inference(model, evidence_dict):
    """
    Performs probabilistic inference on the Manuscript Bayesian Network.
    Args:
        model: The pgmpy Bayesian Network model.
        evidence_dict: Dictionary where keys are variable names and values
                       are their observed states (using integer mapping).
                       e.g., {'Forced_Entry': 1, 'Fingerprints': 3}
    Returns:
        A dictionary containing the posterior probability distribution(s)
        for the query variable(s), or None if inference fails.
    """
    if not evidence_dict:
        print("Performing inference with no evidence (calculating priors).")
    else:
        # Ensure only valid evidence states are passed (e.g., 0 or 1 for binary)
        # And filter out None values which mean 'Unknown' / no evidence for that var
        valid_evidence = {k: int(v) for k, v in evidence_dict.items()
                          if v is not None and k in model.nodes()}
        print(f"Performing inference with evidence: {valid_evidence}")
        if len(valid_evidence) != len({k:v for k,v in evidence_dict.items() if v is not None}):
            print("Warning: Some provided evidence keys were not found in the model nodes.")

    try:
        inference_engine = VariableElimination(model)
    except Exception as e:
        st.error(f"Failed to initialize inference engine: {e}")
        print(f"Failed to initialize inference engine: {e}\n{traceback.format_exc()}")
        return None

    query_variables = ['Culprit'] # We want the posterior probability of the Culprit node

    try:
        # Perform the query
        results = inference_engine.query(variables=query_variables, evidence=valid_evidence)
        print("Inference successful.")
        # pgmpy query returns a factor; we'll process it later
        return results # Return the factor directly

    except Exception as e:
        st.error(f"Inference failed: {e}")
        print(f"Inference failed: {e}\n{traceback.format_exc()}")
        # Provide more debugging info if possible (may vary with pgmpy versions/error types)
        if 'factors' in str(e).lower() or hasattr(e, 'factors'):
             print(f"Potential factors involved in error (if available): {getattr(e, 'factors', 'N/A')}")
        st.error("Check if evidence contradicts CPTs (e.g., P=0 situations). Review CPT definitions.")
        return None

# --- Helper function to create Graphviz object ---
def create_graphviz_object(model):
    """Creates a graphviz.Digraph object from the pgmpy model."""
    dot = graphviz.Digraph(comment='Manuscript Bayesian Network')
    dot.attr(rankdir='TB', size='8,5') # Top-to-Bottom layout, adjust size if needed

    # Add nodes with labels
    for node in model.nodes():
        # Make node labels more descriptive if needed, or just use variable name
        dot.node(node, node.replace('_', ' '))

    # Add edges
    for edge in model.edges():
        dot.edge(edge[0], edge[1])

    return dot

# --- Streamlit User Interface ---

st.set_page_config(page_title="Manuscript Mystery Solver", layout="wide")

st.title("📜 The Case of the Missing Manuscript")
st.markdown("""
Welcome, Investigator! Use this Bayesian Network tool to analyze evidence in the theft of the rare manuscript.
Input the clues you've gathered, and the system will infer the probability of each suspect being the culprit.

**Suspects:**
*   **Scholar (A):** Had access, potential motive (rivalry).
*   **Butler (B):** Had keys, claims innocence.
*   **Cat Burglar (C):** Professional thief, known to operate in the area.

**How it works:** Based on the dependencies modeled (see graph below) and the defined probabilities (CPTs), the system uses **Bayesian Inference** ($P(\text{Culprit} | \text{Evidence})$) to update beliefs.
""")

# --- Build the Bayesian Network ---
try:
    bn_model_manuscript = build_manuscript_bayesian_network()
    st.success("Bayesian Network for the Manuscript case loaded successfully!")

    # --- UI Elements for Evidence Input ---
    st.sidebar.header("Enter Evidence:")
    evidence_values = {} # Dictionary to store user selections

    # Use descriptive labels for the UI
    evidence_ui_map = {
        'Forced_Entry': "Was there Forced Entry?",
        'Alibi_A': "Does the Scholar (A) have an Alibi?",
        'Alibi_B': "Does the Butler (B) have an Alibi?",
        'Alibi_C': "Does the Cat Burglar (C) have an Alibi?",
        'Fingerprints': "Whose Fingerprints were Found?"
        # Add Motive/Security Footage here if they were added as root nodes
    }

    # Define options for dropdowns
    binary_options = ["Unknown", "No", "Yes"]
    fingerprint_options = ["Unknown", "None", "Scholar (A)", "Butler (B)", "Cat Burglar (C)"]

    # Create widgets for each evidence variable in the model
    for var, label in evidence_ui_map.items():
        if var in bn_model_manuscript.nodes():
            if var == 'Fingerprints':
                user_input = st.sidebar.selectbox(label, options=fingerprint_options, index=0, key=var)
                if user_input != "Unknown":
                    evidence_values[var] = FINGERPRINT_MAP[user_input] # Map string to int
                else:
                    evidence_values[var] = None # None signifies no evidence for this var
            else: # Binary variables (Yes/No)
                user_input = st.sidebar.selectbox(label, options=binary_options, index=0, key=var)
                if user_input != "Unknown":
                    evidence_values[var] = BINARY_MAP[user_input] # Map string to int (0 or 1)
                else:
                    evidence_values[var] = None
        else:
             st.sidebar.warning(f"Variable '{var}' defined in UI map but not in the BN model.")

    # Prepare evidence dict for the inference function (filter out None values)
    evidence_for_inference = {k: v for k, v in evidence_values.items() if v is not None}

    st.header("Evidence Input:")
    if not evidence_for_inference:
        st.info("No specific evidence entered. Showing prior probabilities for the culprit.")
    else:
        # Display evidence in a user-friendly way
        display_evidence = {}
        for k, v in evidence_for_inference.items():
            if k == 'Fingerprints':
                # Find the key (string name) corresponding to the value (int)
                 fp_key = next((name for name, index in FINGERPRINT_MAP.items() if index == v), str(v))
                 display_evidence[evidence_ui_map[k]] = fp_key
            else:
                 # Find the key (string name) corresponding to the value (int) for binary
                 bin_key = next((name for name, index in BINARY_MAP.items() if index == v), str(v))
                 display_evidence[evidence_ui_map[k]] = bin_key
        st.json(display_evidence)


    # --- Perform Inference and Display Results ---
    if st.button("Analyze Evidence / Solve Mystery"):
        st.header("Inference Results: Who is the Culprit?")
        with st.spinner("Calculating updated probabilities..."):
            culprit_posterior_factor = perform_manuscript_inference(bn_model_manuscript, evidence_for_inference)

            if culprit_posterior_factor:
                try:
                    # Extract probabilities for each culprit state
                    prob_none = culprit_posterior_factor.get_value(**{'Culprit': CULPRIT_MAP['None']})
                    prob_a = culprit_posterior_factor.get_value(**{'Culprit': CULPRIT_MAP['Scholar (A)']})
                    prob_b = culprit_posterior_factor.get_value(**{'Culprit': CULPRIT_MAP['Butler (B)']})
                    prob_c = culprit_posterior_factor.get_value(**{'Culprit': CULPRIT_MAP['Cat Burglar (C)']})

                    # Display using columns and metrics for better layout
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric(label="Probability Scholar (A) is Guilty", value=f"{prob_a:.2%}")
                    with col2:
                        st.metric(label="Probability Butler (B) is Guilty", value=f"{prob_b:.2%}")
                    with col3:
                        st.metric(label="Probability Cat Burglar (C) is Guilty", value=f"{prob_c:.2%}")
                    with col4:
                         st.metric(label="Probability No One Guilty / Other", value=f"{prob_none:.2%}")

                    # Optional: Display the raw factor
                    with st.expander("Show Full Posterior Distribution (Factor)"):
                        st.text(f"P(Culprit | Evidence):\n{culprit_posterior_factor}")

                except Exception as e:
                    st.error(f"Error processing inference results: {e}")
                    print(f"Error extracting probabilities from factor: {e}\nFactor:\n{culprit_posterior_factor}")
                    st.error("Could not display probabilities. Check the factor output above.")
            else:
                # Error message already shown by perform_manuscript_inference
                st.warning("Inference failed. Cannot display results.")

    # --- Display Model Information ---
    st.markdown("---")
    st.header("Underlying Bayesian Network Model")

    # Visualization
    st.subheader("Network Graph")
    st.info("Visual representation of the dependencies between variables.")
    try:
        graph_obj = create_graphviz_object(bn_model_manuscript)
        st.graphviz_chart(graph_obj)
    except ImportError:
        st.warning("Python `graphviz` package not found. Please install it (`pip install graphviz`). Graph cannot be rendered.")
    except graphviz.backend.execute.ExecutableNotFound:
        st.error("Graphviz executable not found. Please install Graphviz system software (from graphviz.org) and ensure it's in your system's PATH. Graph cannot be rendered.")
    except Exception as e:
         st.error(f"Could not render graph: {e}")
         print(f"Error rendering graph: {e}\n{traceback.format_exc()}")

    # Structure and Parameters
    col_struct, col_params = st.columns(2)
    with col_struct:
        with st.expander("Show Network Structure (Edges)"):
            st.write(bn_model_manuscript.edges())
    with col_params:
        with st.expander("Show Conditional Probability Distributions (CPDs)"):
            for cpd in bn_model_manuscript.get_cpds():
                st.text(str(cpd))

except ValueError as ve:
    st.error(f"Failed to build the Bayesian Network: {ve}")
    st.error("Please check the CPD definitions and network structure in the `build_manuscript_bayesian_network` function.")
    print(f"Critical error during BN setup: {ve}\n{traceback.format_exc()}")
except Exception as e:
    st.error(f"An unexpected error occurred during application setup: {e}")
    print(f"Critical error during setup: {e}\n{traceback.format_exc()}")