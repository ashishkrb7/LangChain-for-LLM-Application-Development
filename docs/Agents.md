# LangChain: Agents

## Outline:

* Using built in LangChain tools: DuckDuckGo search and Wikipedia
* Defining your own tools


```python
import os

from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import AzureChatOpenAI
_ = load_dotenv(find_dotenv()) # read local .env file

import warnings
warnings.filterwarnings("ignore")
```


```python
# Set OpenAI API key
os.environ["OPENAI_API_TYPE"] = os.getenv("api_type")
os.environ["OPENAI_API_BASE"] = os.getenv("api_base")
os.environ["OPENAI_API_VERSION"] = os.getenv("api_version")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
```


```python
llm = AzureChatOpenAI(deployment_name="chatgpt-gpt35-turbo",model_name="gpt-35-turbo",temperature=0.0)
```

## Built-in LangChain tools


```python
#!pip install -U wikipedia
```


```python
from langchain.agents.agent_toolkits import create_python_agent
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL
from langchain.chat_models import ChatOpenAI
```


```python
tools = load_tools(["llm-math","wikipedia"], llm=llm)
```


```python
agent= initialize_agent(
    tools, 
    llm, 
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose = True)
```


```python
agent("What is the 25% of 300?")
```

    Error in on_chain_start callback: 'name'
    

    [32;1m[1;3mThought: I need to use a calculator to find the answer to this math problem.
    
    Action:
    ```
    {
      "action": "Calculator",
      "action_input": "300*0.25"
    }
    ```
    
    [0m
    Observation: [36;1m[1;3mAnswer: 75.0[0m
    Thought:[32;1m[1;3mI have successfully calculated the answer to the math problem.
    
    Final Answer: 75.0[0m
    
    [1m> Finished chain.[0m
    




    {'input': 'What is the 25% of 300?', 'output': '75.0'}




```python
question = "Tom M. Mitchell is an American computer scientist \
and the Founders University Professor at Carnegie Mellon University (CMU)\
what book did he write?"
result = agent(question) 
```

    Error in on_chain_start callback: 'name'
    

    [32;1m[1;3mThought: I should use Wikipedia to find the answer to this question.
    
    Action:
    ```
    {
      "action": "Wikipedia",
      "action_input": "Tom M. Mitchell"
    }
    ```
    
    [0m
    Observation: [33;1m[1;3mPage: Tom M. Mitchell
    Summary: Tom Michael Mitchell (born August 9, 1951) is an American computer scientist and the Founders University Professor at Carnegie Mellon University (CMU). He is a founder and former Chair of the Machine Learning Department at CMU. Mitchell is known for his contributions to the advancement of machine learning, artificial intelligence, and cognitive neuroscience and is the author of the textbook Machine Learning. He is a member of the United States National Academy of Engineering since 2010. He is also a Fellow of the American Academy of Arts and Sciences, the American Association for the Advancement of Science and a Fellow and past President of the Association for the Advancement of Artificial Intelligence. In October 2018, Mitchell was appointed as the Interim Dean of the School of Computer Science at Carnegie Mellon.
    
    Page: Tom Mitchell (Australian footballer)
    Summary: Thomas Mitchell (born 31 May 1993) is a professional Australian rules footballer playing for the Collingwood Football Club in the Australian Football League (AFL). He previously played for the Sydney Swans from 2012 to 2016, and the Hawthorn Football Club between 2017 and 2022. Mitchell won the Brownlow Medal as the league's best and fairest player in 2018 and set the record for the most disposals in a VFL/AFL match, accruing 54 in a game against Collingwood during that season.[0m
    Thought:[32;1m[1;3mThe book that Tom M. Mitchell wrote is "Machine Learning".
    
    Action:
    ```
    {
      "action": "Wikipedia",
      "action_input": "Machine Learning (book)"
    }
    ```
    
    [0m
    Observation: [33;1m[1;3mPage: Machine learning
    Summary: Machine learning (ML) is a branch of artificial intelligence that leverages data to improve computer performance by giving machines the ability to "learn".Machine learning algorithms build a model based on sample data, known as training data, in order to make predictions or decisions without being explicitly programmed to do so. Machine learning algorithms are used in a wide variety of applications, such as in medicine, email filtering, speech recognition, agriculture, and computer vision, where it is difficult or unfeasible to develop conventional algorithms to perform the needed tasks.A subset of machine learning is closely related to computational statistics, which focuses on making predictions using computers, but not all machine learning is statistical learning. The study of mathematical optimization delivers methods, theory and application domains to the field of machine learning. Data mining is a related field of study, focusing on exploratory data analysis through unsupervised learning.Some implementations of machine learning use data and artificial neural networks in a way that mimics the working of a biological brain.In its application across business problems, machine learning is also referred to as predictive analytics.
    
    Page: Quantum machine learning
    Summary: Quantum machine learning is the integration of quantum algorithms within machine learning programs. The most common use of the term refers to machine learning algorithms for the analysis of classical data executed on a quantum computer, i.e. quantum-enhanced machine learning. While machine learning algorithms are used to compute immense quantities of data, quantum machine learning utilizes qubits and quantum operations or specialized quantum systems to improve computational speed and data storage done by algorithms in a program. This includes hybrid methods that involve both classical and quantum processing, where computationally difficult subroutines are outsourced to a quantum device. These routines can be more complex in nature and executed faster on a quantum computer. Furthermore, quantum algorithms can be used to analyze quantum states instead of classical data. Beyond quantum computing, the term "quantum machine learning" is also associated with classical machine learning methods applied to data generated from quantum experiments (i.e. machine learning of quantum systems), such as learning the phase transitions of a quantum system or creating new quantum experiments. Quantum machine learning also extends to a branch of research that explores methodological and structural similarities between certain physical systems and learning systems, in particular neural networks. For example, some mathematical and numerical techniques from quantum physics are applicable to classical deep learning and vice versa. Furthermore, researchers investigate more abstract notions of learning theory with respect to quantum information, sometimes referred to as "quantum learning theory".
    
    
    
    Page: Timeline of machine learning
    Summary: This page is a timeline of machine learning. Major discoveries, achievements, milestones and other major events in machine learning are included.[0m
    Thought:[32;1m[1;3mTom M. Mitchell wrote the book "Machine Learning".
    
    Final Answer: "Machine Learning".[0m
    
    [1m> Finished chain.[0m
    

## Python Agent


```python
agent = create_python_agent(
    llm,
    tool=PythonREPLTool(),
    verbose=True
)
```


```python
customer_list = [["Harrison", "Chase"], 
                 ["Lang", "Chain"],
                 ["Dolly", "Too"],
                 ["Elle", "Elem"], 
                 ["Geoff","Fusion"], 
                 ["Trance","Former"],
                 ["Jen","Ayai"]
                ]
```


```python
agent.run(f"""Sort these customers by \
last name and then first name \
and print the output: {customer_list}""") 
```

    Error in on_chain_start callback: 'name'
    

    [32;1m[1;3mI can use the sorted() function to sort the list of customers by last name and then first name. I will need to provide a key function to sorted() that returns a tuple of the last name and first name in that order.
    Action: Python REPL
    Action Input:
    ```
    customers = [['Harrison', 'Chase'], ['Lang', 'Chain'], ['Dolly', 'Too'], ['Elle', 'Elem'], ['Geoff', 'Fusion'], ['Trance', 'Former'], ['Jen', 'Ayai']]
    sorted_customers = sorted(customers, key=lambda x: (x[1], x[0]))
    for customer in sorted_customers:
        print(customer)
    ```[0m
    Observation: [36;1m[1;3m['Jen', 'Ayai']
    ['Lang', 'Chain']
    ['Harrison', 'Chase']
    ['Elle', 'Elem']
    ['Trance', 'Former']
    ['Geoff', 'Fusion']
    ['Dolly', 'Too']
    [0m
    Thought:[32;1m[1;3mThe customers have been sorted by last name and then first name, and the output has been printed. 
    Final Answer: [['Jen', 'Ayai'], ['Lang', 'Chain'], ['Harrison', 'Chase'], ['Elle', 'Elem'], ['Trance', 'Former'], ['Geoff', 'Fusion'], ['Dolly', 'Too']][0m
    
    [1m> Finished chain.[0m
    




    "[['Jen', 'Ayai'], ['Lang', 'Chain'], ['Harrison', 'Chase'], ['Elle', 'Elem'], ['Trance', 'Former'], ['Geoff', 'Fusion'], ['Dolly', 'Too']]"



#### View detailed outputs of the chains


```python
import langchain
langchain.debug=True
agent.run(f"""Sort these customers by \
last name and then first name \
and print the output: {customer_list}""") 
langchain.debug=False
```

    [32;1m[1;3m[chain/start][0m [1m[1:chain:AgentExecutor] Entering Chain run with input:
    [0m{
      "input": "Sort these customers by last name and then first name and print the output: [['Harrison', 'Chase'], ['Lang', 'Chain'], ['Dolly', 'Too'], ['Elle', 'Elem'], ['Geoff', 'Fusion'], ['Trance', 'Former'], ['Jen', 'Ayai']]"
    }
    [32;1m[1;3m[chain/start][0m [1m[1:chain:AgentExecutor > 2:chain:LLMChain] Entering Chain run with input:
    [0m{
      "input": "Sort these customers by last name and then first name and print the output: [['Harrison', 'Chase'], ['Lang', 'Chain'], ['Dolly', 'Too'], ['Elle', 'Elem'], ['Geoff', 'Fusion'], ['Trance', 'Former'], ['Jen', 'Ayai']]",
      "agent_scratchpad": "",
      "stop": [
        "\nObservation:",
        "\n\tObservation:"
      ]
    }
    [32;1m[1;3m[llm/start][0m [1m[1:chain:AgentExecutor > 2:chain:LLMChain > 3:llm:AzureChatOpenAI] Entering LLM run with input:
    [0m{
      "prompts": [
        "Human: You are an agent designed to write and execute python code to answer questions.\nYou have access to a python REPL, which you can use to execute python code.\nIf you get an error, debug your code and try again.\nOnly use the output of your code to answer the question. \nYou might know the answer without running any code, but you should still run the code to get the answer.\nIf it does not seem like you can write code to answer the question, just return \"I don't know\" as the answer.\n\n\nPython REPL: A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.\n\nUse the following format:\n\nQuestion: the input question you must answer\nThought: you should always think about what to do\nAction: the action to take, should be one of [Python REPL]\nAction Input: the input to the action\nObservation: the result of the action\n... (this Thought/Action/Action Input/Observation can repeat N times)\nThought: I now know the final answer\nFinal Answer: the final answer to the original input question\n\nBegin!\n\nQuestion: Sort these customers by last name and then first name and print the output: [['Harrison', 'Chase'], ['Lang', 'Chain'], ['Dolly', 'Too'], ['Elle', 'Elem'], ['Geoff', 'Fusion'], ['Trance', 'Former'], ['Jen', 'Ayai']]\nThought:"
      ]
    }
    [36;1m[1;3m[llm/end][0m [1m[1:chain:AgentExecutor > 2:chain:LLMChain > 3:llm:AzureChatOpenAI] [4.32s] Exiting LLM run with output:
    [0m{
      "generations": [
        [
          {
            "text": "I can use the sorted() function to sort the list of customers by last name and then first name. I will need to provide a key function to sorted() that returns a tuple of the last name and first name in that order.\nAction: Python REPL\nAction Input:\n```\ncustomers = [['Harrison', 'Chase'], ['Lang', 'Chain'], ['Dolly', 'Too'], ['Elle', 'Elem'], ['Geoff', 'Fusion'], ['Trance', 'Former'], ['Jen', 'Ayai']]\nsorted_customers = sorted(customers, key=lambda x: (x[1], x[0]))\nfor customer in sorted_customers:\n    print(customer)\n```",
            "generation_info": null,
            "message": {
              "content": "I can use the sorted() function to sort the list of customers by last name and then first name. I will need to provide a key function to sorted() that returns a tuple of the last name and first name in that order.\nAction: Python REPL\nAction Input:\n```\ncustomers = [['Harrison', 'Chase'], ['Lang', 'Chain'], ['Dolly', 'Too'], ['Elle', 'Elem'], ['Geoff', 'Fusion'], ['Trance', 'Former'], ['Jen', 'Ayai']]\nsorted_customers = sorted(customers, key=lambda x: (x[1], x[0]))\nfor customer in sorted_customers:\n    print(customer)\n```",
              "additional_kwargs": {},
              "example": false
            }
          }
        ]
      ],
      "llm_output": {
        "token_usage": {
          "completion_tokens": 144,
          "prompt_tokens": 327,
          "total_tokens": 471
        },
        "model_name": "gpt-35-turbo"
      },
      "run": null
    }
    [36;1m[1;3m[chain/end][0m [1m[1:chain:AgentExecutor > 2:chain:LLMChain] [4.32s] Exiting Chain run with output:
    [0m{
      "text": "I can use the sorted() function to sort the list of customers by last name and then first name. I will need to provide a key function to sorted() that returns a tuple of the last name and first name in that order.\nAction: Python REPL\nAction Input:\n```\ncustomers = [['Harrison', 'Chase'], ['Lang', 'Chain'], ['Dolly', 'Too'], ['Elle', 'Elem'], ['Geoff', 'Fusion'], ['Trance', 'Former'], ['Jen', 'Ayai']]\nsorted_customers = sorted(customers, key=lambda x: (x[1], x[0]))\nfor customer in sorted_customers:\n    print(customer)\n```"
    }
    [32;1m[1;3m[tool/start][0m [1m[1:chain:AgentExecutor > 4:tool:Python REPL] Entering Tool run with input:
    [0m"```
    customers = [['Harrison', 'Chase'], ['Lang', 'Chain'], ['Dolly', 'Too'], ['Elle', 'Elem'], ['Geoff', 'Fusion'], ['Trance', 'Former'], ['Jen', 'Ayai']]
    sorted_customers = sorted(customers, key=lambda x: (x[1], x[0]))
    for customer in sorted_customers:
        print(customer)
    ```"
    [36;1m[1;3m[tool/end][0m [1m[1:chain:AgentExecutor > 4:tool:Python REPL] [0.467ms] Exiting Tool run with output:
    [0m"['Jen', 'Ayai']
    ['Lang', 'Chain']
    ['Harrison', 'Chase']
    ['Elle', 'Elem']
    ['Trance', 'Former']
    ['Geoff', 'Fusion']
    ['Dolly', 'Too']"
    [32;1m[1;3m[chain/start][0m [1m[1:chain:AgentExecutor > 5:chain:LLMChain] Entering Chain run with input:
    [0m{
      "input": "Sort these customers by last name and then first name and print the output: [['Harrison', 'Chase'], ['Lang', 'Chain'], ['Dolly', 'Too'], ['Elle', 'Elem'], ['Geoff', 'Fusion'], ['Trance', 'Former'], ['Jen', 'Ayai']]",
      "agent_scratchpad": "I can use the sorted() function to sort the list of customers by last name and then first name. I will need to provide a key function to sorted() that returns a tuple of the last name and first name in that order.\nAction: Python REPL\nAction Input:\n```\ncustomers = [['Harrison', 'Chase'], ['Lang', 'Chain'], ['Dolly', 'Too'], ['Elle', 'Elem'], ['Geoff', 'Fusion'], ['Trance', 'Former'], ['Jen', 'Ayai']]\nsorted_customers = sorted(customers, key=lambda x: (x[1], x[0]))\nfor customer in sorted_customers:\n    print(customer)\n```\nObservation: ['Jen', 'Ayai']\n['Lang', 'Chain']\n['Harrison', 'Chase']\n['Elle', 'Elem']\n['Trance', 'Former']\n['Geoff', 'Fusion']\n['Dolly', 'Too']\n\nThought:",
      "stop": [
        "\nObservation:",
        "\n\tObservation:"
      ]
    }
    [32;1m[1;3m[llm/start][0m [1m[1:chain:AgentExecutor > 5:chain:LLMChain > 6:llm:AzureChatOpenAI] Entering LLM run with input:
    [0m{
      "prompts": [
        "Human: You are an agent designed to write and execute python code to answer questions.\nYou have access to a python REPL, which you can use to execute python code.\nIf you get an error, debug your code and try again.\nOnly use the output of your code to answer the question. \nYou might know the answer without running any code, but you should still run the code to get the answer.\nIf it does not seem like you can write code to answer the question, just return \"I don't know\" as the answer.\n\n\nPython REPL: A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.\n\nUse the following format:\n\nQuestion: the input question you must answer\nThought: you should always think about what to do\nAction: the action to take, should be one of [Python REPL]\nAction Input: the input to the action\nObservation: the result of the action\n... (this Thought/Action/Action Input/Observation can repeat N times)\nThought: I now know the final answer\nFinal Answer: the final answer to the original input question\n\nBegin!\n\nQuestion: Sort these customers by last name and then first name and print the output: [['Harrison', 'Chase'], ['Lang', 'Chain'], ['Dolly', 'Too'], ['Elle', 'Elem'], ['Geoff', 'Fusion'], ['Trance', 'Former'], ['Jen', 'Ayai']]\nThought:I can use the sorted() function to sort the list of customers by last name and then first name. I will need to provide a key function to sorted() that returns a tuple of the last name and first name in that order.\nAction: Python REPL\nAction Input:\n```\ncustomers = [['Harrison', 'Chase'], ['Lang', 'Chain'], ['Dolly', 'Too'], ['Elle', 'Elem'], ['Geoff', 'Fusion'], ['Trance', 'Former'], ['Jen', 'Ayai']]\nsorted_customers = sorted(customers, key=lambda x: (x[1], x[0]))\nfor customer in sorted_customers:\n    print(customer)\n```\nObservation: ['Jen', 'Ayai']\n['Lang', 'Chain']\n['Harrison', 'Chase']\n['Elle', 'Elem']\n['Trance', 'Former']\n['Geoff', 'Fusion']\n['Dolly', 'Too']\n\nThought:"
      ]
    }
    [36;1m[1;3m[llm/end][0m [1m[1:chain:AgentExecutor > 5:chain:LLMChain > 6:llm:AzureChatOpenAI] [2.27s] Exiting LLM run with output:
    [0m{
      "generations": [
        [
          {
            "text": "The customers have been sorted by last name and then first name.\nFinal Answer: [['Jen', 'Ayai'], ['Lang', 'Chain'], ['Harrison', 'Chase'], ['Elle', 'Elem'], ['Trance', 'Former'], ['Geoff', 'Fusion'], ['Dolly', 'Too']]",
            "generation_info": null,
            "message": {
              "content": "The customers have been sorted by last name and then first name.\nFinal Answer: [['Jen', 'Ayai'], ['Lang', 'Chain'], ['Harrison', 'Chase'], ['Elle', 'Elem'], ['Trance', 'Former'], ['Geoff', 'Fusion'], ['Dolly', 'Too']]",
              "additional_kwargs": {},
              "example": false
            }
          }
        ]
      ],
      "llm_output": {
        "token_usage": {
          "completion_tokens": 67,
          "prompt_tokens": 526,
          "total_tokens": 593
        },
        "model_name": "gpt-35-turbo"
      },
      "run": null
    }
    [36;1m[1;3m[chain/end][0m [1m[1:chain:AgentExecutor > 5:chain:LLMChain] [2.27s] Exiting Chain run with output:
    [0m{
      "text": "The customers have been sorted by last name and then first name.\nFinal Answer: [['Jen', 'Ayai'], ['Lang', 'Chain'], ['Harrison', 'Chase'], ['Elle', 'Elem'], ['Trance', 'Former'], ['Geoff', 'Fusion'], ['Dolly', 'Too']]"
    }
    [36;1m[1;3m[chain/end][0m [1m[1:chain:AgentExecutor] [6.60s] Exiting Chain run with output:
    [0m{
      "output": "[['Jen', 'Ayai'], ['Lang', 'Chain'], ['Harrison', 'Chase'], ['Elle', 'Elem'], ['Trance', 'Former'], ['Geoff', 'Fusion'], ['Dolly', 'Too']]"
    }
    

## Define your own tool


```python
# !pip install DateTime
```


```python
from langchain.agents import tool
from datetime import date
```


```python
@tool
def time(text: str) -> str:
    """Returns todays date, use this for any \
    questions related to knowing todays date. \
    The input should always be an empty string, \
    and this function will always return todays \
    date - any date mathmatics should occur \
    outside this function."""
    return str(date.today())
```


```python
agent= initialize_agent(
    tools + [time], 
    llm, 
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose = True)
```

**Note**: 

The agent will sometimes come to the wrong conclusion (agents are a work in progress!). 

If it does, please try running it again.


```python
try:
    result = agent("whats the date today?") 
except: 
    print("exception on external access")
```

    Error in on_chain_start callback: 'name'
    

    [32;1m[1;3mQuestion: whats the date today?
    Thought: I can use the `time` tool to get today's date
    Action:
    ```
    {
      "action": "time",
      "action_input": ""
    }
    ```
    [0m
    Observation: [38;5;200m[1;3m2023-07-02[0m
    Thought:[32;1m[1;3mI have successfully used the `time` tool to get today's date.
    Final Answer: Today's date is 2023-07-02.[0m
    
    [1m> Finished chain.[0m
    
