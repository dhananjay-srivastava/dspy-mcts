# dspy-mcts

## Overview
This repository contains an implementation of Monte Carlo Tree Search (MCTS) for detecting medical errors in given contexts. The algorithm uses DSPy to provide answers and generate search queries to aid in finding solutions.

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Classes and Methods](#classes-and-methods)
4. [MCTS Algorithm](#mcts-algorithm)
5. [License](#license)
6. [Contributing](#contributing)
7. [Contact](#contact)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/dhananjay-srivastava/dspy-mcts.git
   cd dspy-mcts
   ```
2. Pygraphviz needs to be installed and checked working before running the code, here's what works on colab
   ```bash
   !apt install libgraphviz-dev
   !pip install pygraphviz
   pip install -r requirements.txt
   ```

## Usage
To use this project, you can run the `main.py` script which demonstrates how to use the MCTS implementation for detecting medical errors and generating search queries.

### Running the Script
1. Set your OpenAI API key in the `main.py` script:
   ```python
   os.environ['OPENAI_API_KEY'] = "your_openai_api_key"
   ```
2. Execute the script:
   ```bash
   python main.py
   ```

### Example
The `main.py` script provides an example of how to use the `SimplifiedMCTS` class to analyze a medical context and question. The script:
1. Initializes the MCTS object.
2. Defines a medical context and question.
3. Runs the MCTS algorithm to get the response.
4. Visualizes the MCTS exploration path using a graph.

Here is a brief excerpt from `main.py`:
```python
# Set the OpenAI API key
os.environ['OPENAI_API_KEY'] = "your_openai_api_key"

# Initialize the MCTS object
mcts_obj = SimplifiedMCTS()

# Define the context and question
context = "A 53-year-old man comes to the physician because of a 1-day history of fever and chills, severe malaise, and cough with yellow-green sputum. He works as a commercial fisherman on Lake Superior. Current medications include metoprolol and warfarin. His temperature is 38.5 C (101.3 F), pulse is 96/min, respirations are 26/min, and blood pressure is 98/62 mm Hg. Examination shows increased fremitus and bronchial breath sounds over the right middle lung field. An x-ray of the chest shows consolidation of the right upper lobe. The causal pathogen is Streptococcus pneumoniae."
question = "The causal pathogen is Streptococcus pneumoniae."

response = mcts_obj(context, question)
```

### Visualizing the MCTS Tree
The script includes functions to visualize the MCTS tree using Plotly and NetworkX. The `display_graph` function takes the nodes of the MCTS tree and generates an interactive visualization.

## Classes and Methods

### GenerateAnswer
**Description:** Answers questions based on context and additional context.
- **Inputs:**
  - `context`: Relevant facts (Premise).
  - `additional_context`: Additional relevant facts.
  - `question`: Checks for medical errors.
- **Output:** `answer`: Yes or No.

### GenerateSearchQuery
**Description:** Generates search queries to help answer questions.
- **Inputs:**
  - `context`: Relevant facts.
  - `additional_context`: Additional relevant facts.
  - `question`: Checks for medical errors.
- **Output:** `query`: Search query.

### Node
**Description:** Represents a node in the MCTS tree.
- **Attributes:** `context`, `additional_context`, `question`, `answer`, `query`, `children`, `parent`, `t`, `n`, `uct`.
- **Methods:** `calc_uct`, `is_root`.

## MCTS Algorithm
The MCTS algorithm involves:
1. **Selection:** Selects the best child node based on UCT values.
2. **Expansion:** Expands the node by adding all possible child nodes.
3. **Simulation:** Simulates the outcome from the expanded node.
4. **Backpropagation:** Updates the nodes with the result of the simulation.

## License
This project is licensed under the MIT License.

## Contributing
Contributions are welcome. Please submit pull requests or open issues for any improvements or bugs.

---

Explanation generated with GPT-4o and rechecked by author
