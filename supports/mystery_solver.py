# mystery_solver.py
# Contains the core Bayesian Network definition and visualization logic.

import graphviz
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD

# --- Bayesian Network Definition ---

def build_bayesian_network():
    """
    Defines the structure and CPDs of the Bayesian Network for the mystery.
    Uses DiscreteBayesianNetwork.

    Returns:
        pgmpy.models.DiscreteBayesianNetwork: The configured Bayesian Network model.
    Raises:
        ValueError: If the model structure or CPDs are invalid.
    """
    # Define network structure: (Parent, Child)
    # GuiltyParty (GP) influences FE, Alibis, FP, SF
    model = DiscreteBayesianNetwork([
        ('GuiltyParty', 'ForcedEntry'),
        ('GuiltyParty', 'AlibiA'),
        ('GuiltyParty', 'AlibiB'),
        ('GuiltyParty', 'AlibiC'),
        ('GuiltyParty', 'Fingerprints'),
        ('GuiltyParty', 'SecurityFootage')
    ])

    # --- Define Conditional Probability Distributions (CPDs) ---

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

# --- Function to create Graphviz object from pgmpy model ---
def create_graphviz_plot(model):
    """
    Creates a graphviz Digraph object representing the BN structure.

    Args:
        model (pgmpy.models.DiscreteBayesianNetwork): The Bayesian network model.

    Returns:
        graphviz.Digraph or None: The graphviz object, or None if graphviz is not installed.
    """
    if not graphviz: # Check if graphviz was imported successfully (imported in main.py)
        print("Warning: Graphviz library not found or not passed correctly.")
        return None # Should not happen if called from main.py which handles import

    dot = graphviz.Digraph(comment='Bayesian Network Structure', graph_attr={'rankdir': 'TB'}) # TB = Top to Bottom layout

    # Add nodes to the graph
    for node in model.nodes():
        dot.node(node, node) # Use node name as both ID and label

    # Add edges to the graph
    for edge in model.edges():
        parent, child = edge
        dot.edge(parent, child)

    return dot