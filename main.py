import os
import dspy
from mcts import SimplifiedMCTS

os.environ['OPENAI_API_KEY'] = ""

turbo = dspy.OpenAI(model='gpt-3.5-turbo',logprobs=True)
colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')

dspy.settings.configure(lm=turbo, rm=colbertv2_wiki17_abstracts)

mcts_obj = SimplifiedMCTS()

context = ""
question = "What is the capital of France?"

response = mcts_obj(context, question)
print(response.answer,response.reasoning)
