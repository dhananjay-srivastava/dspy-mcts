import dspy
import numpy as np
import pandas as pd
from dsp.utils import deduplicate

class GenerateAnswer(dspy.Signature):
    """Based on the primary context (Premise), answer the question (Hypothesis) in a yes or no format. Use the additional context to support your answer"""

    context = dspy.InputField(desc="contains important relevant facts (Premise) about the question.")
    additional_context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField(desc="based upon the context (Premise) and the additional context, is there a medical error (contradiction) in the sentence?")
    answer = dspy.OutputField(desc="yes or no")

class GenerateSearchQuery(dspy.Signature):
    """Write a simple search query that will help answer a complex question."""

    context = dspy.InputField(desc="contains important relevant facts about the question.")
    additional_context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField(desc="given the context, is there a medical error in the sentence?")
    query = dspy.OutputField(desc="generate a search query which can help in answering the question.")


class Node:
    def __init__(self):
        self.context = ""
        self.additional_context = ""

        self.question = ""

        self.answer = ""
        self.answer_probability = 0
        self.answer_reasoning = ""
        self.answer_reasoning_probability = 0

        self.query = ""
        self.query_probability = 0
        self.query_reasoning = ""
        self.query_reasoning_probability = 0

        self.t = 0
        self.n = 0
        self.uct = 0

        self.children = []
        self.parent = None

    @staticmethod
    def calc_uct(n,t, n_parent):
        if n == 0:
            uct = t
        else:
            uct = t / n + np.sqrt((np.log(n_parent)*2) / n)
        return uct


    def is_root(self):
        if self.parent is None:
            return True
        else:
            return False

    def backpropogate(self,t):
        curr_node = self.parent
        while not curr_node.is_root():
            curr_node.t += t
            curr_node.n += 1
            curr_node.uct = self.calc_uct(curr_node.n,curr_node.t,curr_node.parent.n)
            curr_node = curr_node.parent


class Graph:
    def __init__(self):
        self.root = None
        self.leaf_stack = []
        self.visited_nodes = []

    def select_best_node(self):
        best_uct = self.leaf_stack[0].uct
        best_node = self.leaf_stack[0]
        for idx,node in enumerate(self.leaf_stack):
            if node.uct >= best_uct:
                best_uct = node.uct
                best_node = node
        self.leaf_stack.pop(idx)
        self.visited_nodes.append(best_node)
        return best_node

    def get_final_node(self, strategy="answer_probability"):
        nodes = self.visited_nodes + self.leaf_stack
        if strategy == "answer_probability":
            vals = [node.answer_probability for node in nodes]
            max_val = max(vals)
            final_node = [node for node in nodes if node.answer_probability == max_val][0]
        elif strategy == "n":
            n_vals = [node.n for node in nodes]
            max_n = max(n_vals)
            final_node = [node for node in nodes if node.n == max_n][0]
        elif strategy == "uct":
            final_node = self.visited_nodes[-1]
        return final_node


class SimplifiedMCTS(dspy.Module):
    def __init__(self, passages_per_hop=1, max_hops=5, num_children=3):
        super().__init__()

        self.generate_query = [dspy.ChainOfThought(GenerateSearchQuery,n=num_children) for _ in range(max_hops)]
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.generate_answer = [dspy.ChainOfThought(GenerateAnswer) for _  in range(max_hops*num_children)]
        self.max_hops = max_hops
        self.num_children = num_children

    @staticmethod
    def cot_parse_history(lm, n=3, model_name='gpt-3.5-turbo',parse_type='answer'):
        if model_name=='gpt-3.5-turbo':

            output = []

            choices = lm.history[-1]['response']['choices']

            for c in choices:

                reasoning_logprobs = []
                reasoning_tokens = []
                answer_logprobs = []
                answer_tokens = []

                token_and_logprobs = c['logprobs']['content']

                answer_start = False
                for idx,i in enumerate(token_and_logprobs):
                  if answer_start:
                      answer_tokens.append(i['token'])
                      answer_logprobs.append(i['logprob'])
                  else:
                      reasoning_tokens.append(i['token'])
                      reasoning_logprobs.append(i['logprob'])

                  if parse_type=="answer":
                      if i['token']==':' and token_and_logprobs[idx-1]['token'].lower().strip()=='answer':
                          answer_start = True
                  elif parse_type=="query":
                      if i['token']==':' and token_and_logprobs[idx-1]['token'].lower().strip()=='query':
                          answer_start = True
                  else:
                      raise NotImplementedError

                reasoning = ''.join(reasoning_tokens)
                answer = ''.join(answer_tokens)
                reasoning_probability = np.mean([np.exp(i) for i in reasoning_logprobs])
                answer_probability = np.mean([np.exp(i) for i in answer_logprobs])

                if pd.isna(answer_probability):
                    answer_probability = 0
                if pd.isna(reasoning_probability):
                    reasoning_probability = 0

                output.append({'reasoning':reasoning,
                               'answer':answer,
                               'reasoning_probability':reasoning_probability,
                               'answer_probability':answer_probability})
        else:
            raise NotImplementedError
        return output



    def forward(self, context, question):

        g = Graph()
        root = Node()
        root.context = context
        root.question = question

        ans = self.generate_answer[0](context=context,
                                      additional_context="",
                                      question=question)

        answers = self.cot_parse_history(turbo,n=1,parse_type='answer')
        astuff = answers[0]
        root.answer = astuff['answer']
        root.answer_probability = astuff['answer_probability']
        root.answer_reasoning = astuff['reasoning']
        root.answer_reasoning_probability = astuff['reasoning_probability']

        root.t = astuff['answer_probability']
        root.n = 1
        root.uct = astuff['answer_probability']


        g.root = root
        g.leaf_stack.append(root)

        for hop in range(self.max_hops):
            #selection
            node = g.select_best_node()

            #expansion
            query = self.generate_query[hop](context=context, additional_context=node.additional_context,question=question).query
            queries = self.cot_parse_history(turbo, parse_type='query')

            children = []
            for i in range(self.num_children):
                child_node = Node()
                child_node.context = context
                child_node.question = question
                child_node.parent = node

                qstuff = queries[i]
                child_node.query = qstuff['answer']
                child_node.query_probability = qstuff['answer_probability']
                child_node.query_reasoning = qstuff['reasoning']
                child_node.query_reasoning_probability = qstuff['reasoning_probability']


                passage = self.retrieve(child_node.query).passages[0]
                child_node.additional_context += "\n"+ passage

                children.append(child_node)



            #simulation
            for child_idx,child_node in enumerate(children):

                ans = self.generate_answer[((hop-1)*self.num_children)+child_idx](context=child_node.context,
                                                                            additional_context=child_node.additional_context,
                                                                            question=child_node.question)
                answers = self.cot_parse_history(turbo,n=1,parse_type='answer')
                astuff = answers[0]
                child_node.answer = astuff['answer']
                child_node.answer_probability = astuff['answer_probability']
                child_node.answer_reasoning = astuff['reasoning']
                child_node.answer_reasoning_probability = astuff['reasoning_probability']

                t = astuff['answer_probability']

                child_node.t = t
                child_node.n = 1
                child_node.uct = child_node.calc_uct(1,t,child_node.parent.n)

            node.children = children
            g.leaf_stack += children

            #backpropogation
            best_uct = 0
            best_node = None
            for child_node in node.children:
                if child_node.uct > best_uct:
                    best_uct = child_node.uct
                    best_node = child_node

            best_node.backpropogate(best_node.t)

        final_best_node = g.get_final_node()
        return dspy.Prediction(context=final_best_node.context,
                               additional_context=final_best_node.additional_context,
                               answer=final_best_node.answer,
                               answer_probability=final_best_node.answer_probability,
                               answer_reasoning=final_best_node.answer_reasoning,
                               complete_graph=g)