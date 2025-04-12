# mystery_solver_app.py
import streamlit as st
import pandas as pd
# Import DiscreteBayesianNetwork instead of the deprecated BayesianNetwork
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# --- Bayesian Network Definition ---

def build_bayesian_network():
    """
    Defines the structure and CPDs of the Bayesian Network for the mystery.
    Uses DiscreteBayesianNetwork.
    """
    # Define network structure: (Parent, Child)
    # GuiltyParty (GP) influences FE, Alibis, FP, SF
    # Use DiscreteBayesianNetwork here
    model = DiscreteBayesianNetwork([
        ('GuiltyParty', 'ForcedEntry'),
        ('GuiltyParty', 'AlibiA'),
        ('GuiltyParty', 'AlibiB'),
        ('GuiltyParty', 'AlibiC'),
        ('GuiltyParty', 'Fingerprints'),
        ('GuiltyParty', 'SecurityFootage')
    ])

    # --- Define Conditional Probability Distributions (CPDs) ---
    # CPD definitions remain the same

    # P(GuiltyParty) - Prior
    cpd_gp = TabularCPD(variable='GuiltyParty', variable_card=3,
                        values=[[1/3], [1/3], [1/3]],
                        state_names={'GuiltyParty': ['A', 'B', 'C']})

    # P(ForcedEntry | GuiltyParty)
    cpd_fe = TabularCPD(variable='ForcedEntry', variable_card=2,
                        values=[[0.1, 0.1, 0.9],  # FE = Yes
                                [0.9, 0.9, 0.1]], # FE = No
                        evidence=['GuiltyParty'], evidence_card=[3],
                        state_names={'ForcedEntry': ['Yes', 'No'],
                                     'GuiltyParty': ['A', 'B', 'C']})

    # P(AlibiA | GuiltyParty)
    cpd_aa = TabularCPD(variable='AlibiA', variable_card=2,
                        values=[[0.3, 0.8, 0.8],  # AlibiA = Yes
                                [0.7, 0.2, 0.2]], # AlibiA = No
                        evidence=['GuiltyParty'], evidence_card=[3],
                        state_names={'AlibiA': ['Yes', 'No'],
                                     'GuiltyParty': ['A', 'B', 'C']})

    # P(AlibiB | GuiltyParty)
    cpd_ab = TabularCPD(variable='AlibiB', variable_card=2,
                        values=[[0.8, 0.3, 0.7],  # AlibiB = Yes
                                [0.2, 0.7, 0.3]], # AlibiB = No
                        evidence=['GuiltyParty'], evidence_card=[3],
                        state_names={'AlibiB': ['Yes', 'No'],
                                     'GuiltyParty': ['A', 'B', 'C']})

    # P(AlibiC | GuiltyParty)
    cpd_ac = TabularCPD(variable='AlibiC', variable_card=2,
                        values=[[0.7, 0.7, 0.2],  # AlibiC = Yes
                                [0.3, 0.3, 0.8]], # AlibiC = No
                        evidence=['GuiltyParty'], evidence_card=[3],
                        state_names={'AlibiC': ['Yes', 'No'],
                                     'GuiltyParty': ['A', 'B', 'C']})

    # P(Fingerprints | GuiltyParty)
    cpd_fp = TabularCPD(variable='Fingerprints', variable_card=4,
                        values=[[0.4, 0.4, 0.6],  # FP = None
                                [0.5, 0.05, 0.1], # FP = A
                                [0.05, 0.5, 0.1], # FP = B
                                [0.05, 0.05, 0.2]],# FP = C
                        evidence=['GuiltyParty'], evidence_card=[3],
                        state_names={'Fingerprints': ['None', 'A', 'B', 'C'],
                                     'GuiltyParty': ['A', 'B', 'C']})

    # P(SecurityFootage | GuiltyParty)
    cpd_sf = TabularCPD(variable='SecurityFootage', variable_card=2,
                        values=[[0.4, 0.4, 0.2],  # SF = Yes (Useful)
                                [0.6, 0.6, 0.8]], # SF = No (Disabled/Useless)
                        evidence=['GuiltyParty'], evidence_card=[3],
                        state_names={'SecurityFootage': ['Yes', 'No'],
                                     'GuiltyParty': ['A', 'B', 'C']})

    # Add CPDs to the model
    model.add_cpds(cpd_gp, cpd_fe, cpd_aa, cpd_ab, cpd_ac, cpd_fp, cpd_sf)

    # Check model validity - this method works for DiscreteBayesianNetwork too
    if not model.check_model():
        raise ValueError("Model definition is invalid. Check CPDs and structure.")

    return model

# --- Streamlit App Interface ---
# The rest of the Streamlit code remains exactly the same

st.set_page_config(page_title="Bayesian Mystery Solver", layout="wide")

st.title("🕵️‍♂️ Bayesian Mystery Solver")
st.markdown("Using Bayesian Networks to solve 'The Case of the Missing Manuscript'")

st.sidebar.header("Case Scenario")
st.sidebar.markdown("""
A rare, valuable manuscript has been stolen from a locked library room overnight.

**Suspects:**
*   **A (The Scholar):** Had access, known rivalry with the owner.
*   **B (The Butler):** Had keys, claims to have heard nothing.
*   **C (The Cat Burglar):** Known professional thief, operates in the area.
""")

st.sidebar.header("Enter Clues (Evidence)")

# --- Evidence Input Widgets ---
# Using selectbox for clarity, mapping 'Unknown' to None
evidence_options = {'Yes': 'Yes', 'No': 'No', 'Unknown': None}
fp_options = {'None Found': 'None', 'Match Scholar A': 'A', 'Match Butler B': 'B', 'Match Cat Burglar C': 'C', 'Unknown': None}

# Map display text to internal state names
fe_input_display = st.sidebar.selectbox("1. Forced Entry?", options=list(evidence_options.keys()), index=2) # Default Unknown
fe_input = evidence_options[fe_input_display]

m_a_input_display = st.sidebar.selectbox("2. Strong Motive for Scholar A?", options=list(evidence_options.keys()), index=2)
# Note: Motive is treated as evidence affecting likelihood, not a node in this model structure

m_b_input_display = st.sidebar.selectbox("3. Strong Motive for Butler B?", options=list(evidence_options.keys()), index=2)
# Similar note for Motive B

a_a_input_display = st.sidebar.selectbox("4. Alibi for Scholar A?", options=list(evidence_options.keys()), index=2)
a_a_input = evidence_options[a_a_input_display]

a_b_input_display = st.sidebar.selectbox("5. Alibi for Butler B?", options=list(evidence_options.keys()), index=2)
a_b_input = evidence_options[a_b_input_display]

a_c_input_display = st.sidebar.selectbox("6. Alibi for Cat Burglar C?", options=list(evidence_options.keys()), index=2)
a_c_input = evidence_options[a_c_input_display]

fp_input_display = st.sidebar.selectbox("7. Fingerprints Found?", options=list(fp_options.keys()), index=4) # Default Unknown
fp_input = fp_options[fp_input_display]

sf_input_display = st.sidebar.selectbox("8. Useful Security Footage?", options=list(evidence_options.keys()), index=2)
sf_input = evidence_options[sf_input_display]

# Build the Bayesian Network
try:
    model = build_bayesian_network()
    # VariableElimination works with DiscreteBayesianNetwork
    inference = VariableElimination(model)
except ValueError as e:
    st.error(f"Error building the Bayesian Network: {e}")
    st.stop() # Stop execution if model is invalid
except ImportError as e:
    st.error(f"Import Error: {e}. Make sure pgmpy is installed correctly.")
    st.stop()
except Exception as e: # Catch other potential errors during model build
    st.error(f"An unexpected error occurred during model setup: {e}")
    st.stop()

# --- Perform Inference ---
st.header("Inference Results")

solve_button = st.button("Solve Mystery Based on Clues")

if solve_button:
    # Collect evidence dictionary, filtering out None values
    evidence_dict = {}
    if fe_input is not None: evidence_dict['ForcedEntry'] = fe_input
    if a_a_input is not None: evidence_dict['AlibiA'] = a_a_input
    if a_b_input is not None: evidence_dict['AlibiB'] = a_b_input
    if a_c_input is not None: evidence_dict['AlibiC'] = a_c_input
    if fp_input is not None: evidence_dict['Fingerprints'] = fp_input
    if sf_input is not None: evidence_dict['SecurityFootage'] = sf_input

    st.subheader("Evidence Considered:")
    if not evidence_dict:
        st.write("No specific clues entered. Showing prior probabilities.")
    else:
        # Using st.json for cleaner dictionary display
        st.json(evidence_dict)

    # Perform the query
    try:
        # Ensure inference object is valid before querying
        if 'inference' not in locals():
             st.error("Inference engine not initialized. Cannot solve.")
        else:
            posterior_gp = inference.query(variables=['GuiltyParty'], evidence=evidence_dict)

            st.subheader("Posterior Probability of Guilt:")

            # Format results nicely using Pandas DataFrame
            prob_df = pd.DataFrame({
                'Suspect': posterior_gp.state_names['GuiltyParty'],
                'Probability': posterior_gp.values
            })
            prob_df = prob_df.sort_values(by='Probability', ascending=False)
            # Format probability as percentage string for display
            prob_df['Probability'] = prob_df['Probability'].map('{:.2%}'.format)

            st.dataframe(prob_df, use_container_width=True) # Use container width for better layout

            st.subheader("Probability Distribution")
            # Create data suitable for st.bar_chart (index=category, column=value)
            # Use the raw numerical probabilities for the chart
            chart_data = pd.DataFrame(
                posterior_gp.values,
                index=posterior_gp.state_names['GuiltyParty'],
                columns=['Probability']
            )
            st.bar_chart(chart_data)

            # Add interpretation hints for motive evidence (since it's not directly in the BN model)
            st.markdown("---")
            st.markdown("**Interpretation Notes:**")
            st.markdown("*   Remember that the 'Motive' inputs (M_A, M_B) were not directly nodes in this specific network but should influence your interpretation. Strong motive for a suspect with high posterior probability reinforces the suspicion.*")
            st.markdown("*   Probabilities reflect the model's belief based *only* on the specified structure, CPDs, and entered evidence.*")


    except ValueError as e:
        st.error(f"An error occurred during inference: {e}")
        st.warning("This can sometimes happen if the evidence provided strongly contradicts the model's probabilities (e.g., zero probability paths). Try removing some evidence.")
    except Exception as e:
        st.error(f"An unexpected error occurred during inference: {e}")
        st.error("Details: " + str(e))


else:
    st.info("Enter clues in the sidebar and click 'Solve Mystery'.")

    # Show prior probabilities if button not clicked yet
    try:
         # Ensure inference object is valid before querying
        if 'inference' not in locals():
             st.error("Inference engine not initialized. Cannot show priors.")
        else:
            prior_gp = inference.query(variables=['GuiltyParty'])
            st.subheader("Initial (Prior) Probability of Guilt:")
            prior_df = pd.DataFrame({
                'Suspect': prior_gp.state_names['GuiltyParty'],
                'Probability': prior_gp.values
            })
            prior_df['Probability'] = prior_df['Probability'].map('{:.2%}'.format)
            st.dataframe(prior_df, use_container_width=True)

            # Chart for priors
            chart_data = pd.DataFrame(
                prior_gp.values,
                index=prior_gp.state_names['GuiltyParty'],
                columns=['Probability']
                )
            st.bar_chart(chart_data)
    except Exception as e:
        st.error(f"An error occurred while calculating prior probabilities: {e}")


st.markdown("---")
st.markdown("Built using `pgmpy` and `streamlit`.") 