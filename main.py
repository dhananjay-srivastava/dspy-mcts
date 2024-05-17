import os
import dspy
from mcts import SimplifiedMCTS
import plotly.graph_objects as go
import networkx as nx

from collections import deque


def get_all_nodes_bfs(root):
    all_nodes = []
    id_ctr = 0
    queue = deque([(id_ctr,None,root)])


    while queue:
        id,parent_id,node = queue.popleft()
        n = {'id': id,
             'parent': parent_id,
             'answer': node.answer,
             'answer_probability': node.answer_probability,
             'reasoning': node.answer_reasoning,
             'reasoning_probability': node.answer_reasoning_probability,
             'relevant_wiki':node.query,
             'rag_probability':node.query_probability
             }
        all_nodes.append(n)
        for child in node.children:
            id_ctr += 1
            queue.append((id_ctr,id,child))

    return all_nodes

def display_graph(nodes):
    # Create a NetworkX graph
    G = nx.DiGraph()

    # Add nodes and edges to the graph
    for node in nodes:
        G.add_node(node["id"],
                  answer=node["answer"],
                  answer_probability=node["answer_probability"],
                  reasoning=node["reasoning"],
                  reasoning_probability=node["reasoning_probability"],
                  relevant_wiki=node['relevant_wiki'],
                  rag_probability=node['rag_probability'])
        if node["parent"] is not None:
            G.add_edge(node["parent"], node["id"])

    # Generate positions for nodes in a hierarchical layout
    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")

    # Extract the node attributes
    node_x = []
    node_y = []
    node_text = []
    for node_id in G.nodes:
        x, y = pos[node_id]
        node_x.append(x)
        node_y.append(y)
        node_data = G.nodes[node_id]
        hover_text = (f"Answer: {node_data['answer']}<br>"
                      f"Answer Probability: {node_data['answer_probability']}<br>"
                      f"Reasoning: {'<br>'.join(node_data['reasoning'].splitlines())}<br>"
                      f"Reasoning Probability: {node_data['reasoning_probability']}<br>"
                      f"Relevant Wiki: {'<br>'.join(node_data['relevant_wiki'].splitlines())}<br>"
                      f"RAG Probability: {node_data['rag_probability']}<br>")
        node_text.append(hover_text)

    # Extract the edge positions
    edge_x = []
    edge_y = []
    for edge in G.edges:
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    # Create the figure
    fig = go.Figure()

    # Add edges as traces
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='gray'),
        hoverinfo='none',
        mode='lines'))

    # Add nodes as traces
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=[str(node_id) for node_id in G.nodes],
        marker=dict(size=10, color='lightblue'),
        textposition='top center',
        hoverinfo='text',
        hovertext=node_text))

    # Update layout for better visualization
    fig.update_layout(
        title='MCTS Tree exploration path',
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        plot_bgcolor='white',
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig

# Set the OpenAI API key
os.environ['OPENAI_API_KEY'] = ""

# Initialize the MCTS object
mcts_obj = SimplifiedMCTS()

# Define the context and question
context = "A 53-year-old man comes to the physician because of a 1-day history of fever and chills, severe malaise, and cough with yellow-green sputum. He works as a commercial fisherman on Lake Superior. Current medications include metoprolol and warfarin. His temperature is 38.5 C (101.3 F), pulse is 96/min, respirations are 26/min, and blood pressure is 98/62 mm Hg. Examination shows increased fremitus and bronchial breath sounds over the right middle lung field. An x-ray of the chest shows consolidation of the right upper lobe. The causal pathogen is Streptococcus pneumoniae. "
question = "The causal pathogen is Streptococcus pneumoniae."

response = mcts_obj(context, question)

# Sample data: a list of nodes with their attributes and edges
nodes = get_all_nodes_bfs(response.complete_graph.root)

# Display the graph
fig = display_graph(nodes)

# Show the figure
fig.show()