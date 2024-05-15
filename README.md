# MCTS Implementation in DSPy

## Overview

This repository contains an implementation of Monte Carlo Tree Search (MCTS) for detecting medical errors in given contexts. The algorithm is designed to answer questions based on the provided context and additional context, and it can also generate search queries to aid in finding answers.

## Table of Contents

1. [Usage](#usage)
2. [Classes and Methods](#classes-and-methods)
3. [MCTS Algorithm](#mcts-algorithm)

## Usage

You can use the provided classes and methods to perform MCTS-based medical error detection and query generation. The primary classes are `GenerateAnswer` and `GenerateSearchQuery`, which are signatures for the tasks.

## Classes and Methods

### GenerateAnswer
- **Description**: Based on the primary context (Premise), answers the question (Hypothesis) in a yes or no format using additional context to support the answer.
- **Inputs**:
  - `context`: Contains important relevant facts (Premise) about the question.
  - `additional_context`: May contain relevant facts.
  - `question`: Based upon the context (Premise) and the additional context, checks if there is a medical error (contradiction) in the sentence.
- **Output**:
  - `answer`: Yes or No.

### GenerateSearchQuery
- **Description**: Writes a simple search query that will help answer a complex question.
- **Inputs**:
  - `context`: Contains important relevant facts about the question.
  - `additional_context`: May contain relevant facts.
  - `question`: Given the context, checks if there is a medical error in the sentence.
- **Output**:
  - `query`: Generates a search query to help in answering the question.

### Node
- **Description**: Represents a node in the MCTS tree.
- **Attributes**:
  - `context`, `additional_context`, `question`: Stores the relevant data.
  - `answer`, `answer_probability`, `answer_reasoning`, `answer_reasoning_probability`: Stores the answer details.
  - `query`, `query_probability`, `query_reasoning`, `query_reasoning_probability`: Stores the query details.
  - `t`, `n`, `uct`: Stores the UCT values for the node.
  - `children`: List of child nodes.
  - `parent`: Reference to the parent node.
- **Methods**:
  - `calc_uct(n, t, n_parent)`: Calculates the Upper Confidence Bound for Trees (UCT) value.
  - `is_root()`: Checks if the node is the root node.

## MCTS Algorithm

The Monte Carlo Tree Search (MCTS) algorithm involves:
1. **Selection**: Selecting the best child node based on UCT values.
2. **Expansion**: Expanding the node by adding all possible child nodes.
3. **Simulation**: Simulating the outcome from the expanded node.
4. **Backpropagation**: Backpropagating the result of the simulation up the tree to update the nodes.


---

Explanation generated with GPT-4o and rechecked by author
