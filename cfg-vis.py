import glob
import os
import pickle
import random

import networkx as nx

# --- Configuration ---
BENIGN_DIR = "./data/CFG_dataset/Train_CFG/Benign_CFG"
MALWARE_DIR = "./data/CFG_dataset/Train_CFG/Malware_CFG"
OUTPUT_DIR = "./visualized_cfgs"  # Directory to save the output images
NUM_FILES_TO_GRAB = 5

# --- Setup and File Gathering (Remains the same) ---

os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_random_gpickle_files(directory, count):
    """Gathers all gpickle files and returns a random selection."""
    search_path = os.path.join(directory, "*.gpickle")
    all_files = glob.glob(search_path)
    if not all_files:
        print(f"⚠️ Warning: No gpickle files found in {directory}. Skipping.")
        return []
    return random.sample(all_files, min(count, len(all_files)))


files_to_process = []
num_benign = NUM_FILES_TO_GRAB // 2
num_malware = NUM_FILES_TO_GRAB - num_benign

files_to_process.extend(get_random_gpickle_files(BENIGN_DIR, num_benign))
files_to_process.extend(get_random_gpickle_files(MALWARE_DIR, num_malware))

# --- Visualization Logic ---

print(f"✨ Found and selected {len(files_to_process)} CFG files for visualization.")

for file_path in files_to_process:
    base_filename = os.path.basename(file_path).replace(".gpickle", "")
    output_path = os.path.join(OUTPUT_DIR, f"{base_filename}_cfg.png")

    print(f"\nProcessing: {file_path}")
    print(f"Output to: {output_path}")

    try:
        # 1. Load the CFG using pickle
        print("Attempting to load graph with pickle.load...")
        with open(file_path, "rb") as f:
            cfg_graph = pickle.load(f)

        if not isinstance(cfg_graph, nx.Graph):
            print(
                f"⚠️ Warning: Loaded object for {base_filename} is not a standard NetworkX graph."
            )

        # 2. PRE-ANNOTATE THE GRAPH (ESSENTIAL FOR LABELING)
        for node_addr, node_data in cfg_graph.nodes(data=True):
            # Use 'content' for detailed instructions, or fallback to address
            content = node_data.get("content")
            if content:
                label_text = (
                    "\n".join(str(i) for i in content)
                    if isinstance(content, list)
                    else str(content)
                )
            else:
                label_text = (
                    hex(node_addr) if isinstance(node_addr, int) else str(node_addr)
                )

            # Set the standard Graphviz 'label' and 'shape' attributes
            node_data["label"] = label_text
            node_data["shape"] = "box"
            node_data["style"] = "filled"
            node_data["fillcolor"] = "#EDEDED"  # Light gray background

        # 3. CONVERT AND SAVE USING PYDOT (The robust visualization method)
        print("Converting to DOT and rendering with pydot...")

        # Convert the NetworkX graph to a pydot graph object
        pdot_graph = nx.drawing.nx_pydot.to_pydot(cfg_graph)

        # Set Graphviz attributes for the entire graph (layout and direction)
        pdot_graph.set_rankdir("TB")  # Top-to-Bottom flow (typical for CFGs)
        pdot_graph.set_overlap("false")
        pdot_graph.set_splines("true")

        # Save the graph as a PNG file
        pdot_graph.write_png(output_path)

        print(f"✅ Successfully visualized and saved: {output_path}")

    except nx.NetworkXException as ne:
        print(
            f"❌ NetworkX Error: Failed to convert graph to pydot. This might indicate malformed node/edge data. Error: {ne}"
        )
    except Exception as e:
        print(f"❌ Failed to process {file_path}. Error: {e}")

print("\n--- Script Finished ---")
