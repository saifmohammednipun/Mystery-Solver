# App/main.py
# Streamlit application interface for the Bayesian Mystery Solver.

import streamlit as st
import pandas as pd
from pgmpy.inference import VariableElimination

# --- MODIFIED IMPORT ---
# Import functions from our custom module located in the 'supports' subdirectory
try:
    from supports.mystery_solver import build_bayesian_network, create_graphviz_plot
except ModuleNotFoundError:
    st.error("ERROR: Could not find the 'supports/mystery_solver.py' module. Please ensure the file structure is correct (App/supports/mystery_solver.py) and includes an empty '__init__.py' file in the 'supports' directory.")
    st.stop() # Stop execution if core module is missing
except ImportError as e:
     st.error(f"ERROR importing from 'supports.mystery_solver': {e}. Check dependencies within mystery_solver.py.")
     st.stop()


# Try importing graphviz, needed for visualization check
try:
    import graphviz
except ImportError:
    st.warning("Graphviz library not found. Visualization disabled. To enable, install it (`pip install graphviz`) and ensure the Graphviz system tools are installed (see https://graphviz.org/download/).")
    graphviz = None # Set to None if import fails

# --- Streamlit App Interface ---

st.set_page_config(page_title="Bayesian Mystery Solver", layout="wide")

st.title("🕵️‍♂️ ClueChain: A Bayesian Network-Based Mystery Solver Application")
st.markdown("Using Bayesian Networks to solve 'The Case of the Missing Manuscript'")
st.markdown("""---""") # Separator

# --- Build Model and Inference Engine ---
# Encapsulate model building and inference setup
model = None
inference = None
model_built = False
graph_viz_object = None

try:
    # Build the Bayesian Network using the imported function
    model = build_bayesian_network()
    # Create the inference engine instance
    inference = VariableElimination(model)
    model_built = True
    # Create the graphviz object if graphviz is available
    if graphviz and model: # Check model exists too
         graph_viz_object = create_graphviz_plot(model) # Pass the model to the function

except ValueError as e:
    st.error(f"Error building the Bayesian Network: {e}")
except Exception as e: # Catch other potential errors during model setup
    st.error(f"An unexpected error occurred during model setup: {e}")

# --- Display Network Structure (if successful) ---
# Check graph_viz_object specifically, as create_graphviz_plot might return None
if model_built and graph_viz_object:
    st.subheader("Bayesian Network Model")
    st.markdown("""
    This graph shows the variables (nodes) and the assumed direct dependencies (arrows) in our model.
    - An arrow from **G → E** means variable **G** directly influences variable **E**.
    - The *absence* of a path between nodes implies *conditional independence* given certain other nodes. For example, `ForcedEntry` is conditionally independent of `AlibiA` *given* the `GuiltyParty`.
    """)
    try:
        st.graphviz_chart(graph_viz_object)
    except Exception as e:
        st.error(f"Failed to render Graphviz chart: {e}")
    st.markdown("""---""")
elif model_built and not graphviz:
    # This message is now handled by the initial graphviz import check
    # st.warning("Graphviz library not detected. Skipping network visualization.")
    st.markdown("""---""")


# --- Sidebar for Inputs ---
# ... (Sidebar code remains identical to the previous version) ...
st.sidebar.header("Case Scenario")
st.sidebar.markdown("""
A rare, valuable manuscript has been stolen from a locked library room overnight.

**Suspects:**
*   **A (The Scholar):** Had access, known rivalry with the owner.
*   **B (The Butler):** Had keys, claims to have heard nothing.
*   **C (The Cat Burglar):** Known professional thief, operates in the area.
""")

st.sidebar.header("Enter Clues (Evidence)")

# Evidence Input Widgets
evidence_options = {'Yes': 'Yes', 'No': 'No', 'Unknown': None}
fp_options = {'None Found': 'None', 'Match Scholar A': 'A', 'Match Butler B': 'B', 'Match Cat Burglar C': 'C', 'Unknown': None}

fe_input_display = st.sidebar.selectbox("1. Forced Entry?", options=list(evidence_options.keys()), index=2, key="fe")
fe_input = evidence_options[fe_input_display]

m_a_input_display = st.sidebar.selectbox("2. Strong Motive for Scholar A?", options=list(evidence_options.keys()), index=2, key="ma")
# Motive note remains

m_b_input_display = st.sidebar.selectbox("3. Strong Motive for Butler B?", options=list(evidence_options.keys()), index=2, key="mb")
# Motive note remains

a_a_input_display = st.sidebar.selectbox("4. Alibi for Scholar A?", options=list(evidence_options.keys()), index=2, key="aa")
a_a_input = evidence_options[a_a_input_display]

a_b_input_display = st.sidebar.selectbox("5. Alibi for Butler B?", options=list(evidence_options.keys()), index=2, key="ab")
a_b_input = evidence_options[a_b_input_display]

a_c_input_display = st.sidebar.selectbox("6. Alibi for Cat Burglar C?", options=list(evidence_options.keys()), index=2, key="ac")
a_c_input = evidence_options[a_c_input_display]

fp_input_display = st.sidebar.selectbox("7. Fingerprints Found?", options=list(fp_options.keys()), index=4, key="fp")
fp_input = fp_options[fp_input_display]

sf_input_display = st.sidebar.selectbox("8. Useful Security Footage?", options=list(evidence_options.keys()), index=2, key="sf")
sf_input = evidence_options[sf_input_display]

# --- Perform Inference ---
# ... (Inference code remains identical to the previous version) ...
st.header("Inference Results")

# Only show solve button and results if model was built successfully
if model_built:
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
            st.json(evidence_dict)

        # Perform the query using the inference engine
        try:
            # Ensure inference engine exists before querying
            if inference is None:
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
                # Format probability as percentage string for display table
                prob_df['Probability_pct'] = prob_df['Probability'].map('{:.2%}'.format)

                st.dataframe(prob_df[['Suspect', 'Probability_pct']], use_container_width=True)

                st.subheader("Probability Distribution")
                # Create data suitable for st.bar_chart (index=category, column=value)
                # Use the raw numerical probabilities for the chart, sorted
                chart_data = pd.DataFrame(
                    prob_df['Probability'].values, # Use raw probability
                    index=prob_df['Suspect'].values,
                    columns=['Probability']
                ).sort_values('Probability', ascending=False) # Ensure chart matches table sort
                st.bar_chart(chart_data)

                # Interpretation notes (remain the same)
                st.markdown("---")
                st.markdown("**Interpretation Notes:**")
                st.markdown("*   Remember that the 'Motive' inputs (M_A, M_B) were not directly nodes in this specific network but should influence your interpretation...")
                st.markdown("*   Probabilities reflect the model's belief based *only* on the specified structure, CPDs, and entered evidence.*")

        except ValueError as e:
            st.error(f"An error occurred during inference: {e}")
            st.warning("This can sometimes happen if evidence conflicts strongly. Try removing some evidence.")
        except Exception as e:
            st.error(f"An unexpected error occurred during inference: {e}")
            st.error("Details: " + str(e))

    else:
        # Display priors if button hasn't been clicked yet this run
        st.info("Enter clues in the sidebar and click 'Solve Mystery'.")
        try:
            # Ensure inference engine exists before querying
            if inference is None:
                 st.error("Inference engine not initialized. Cannot show priors.")
            else:
                prior_gp = inference.query(variables=['GuiltyParty']) # Query with no evidence
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

else: # If model failed to build initially
    st.warning("The Bayesian Network model could not be built. Please check console/logs for errors. Inference is disabled.")


st.markdown("---")
st.markdown("Built using `pgmpy`, `streamlit`, and `graphviz`.")