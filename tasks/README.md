## Tasks for experiments

Currently a task is a Hugging Face dataset wrapped in its own Python module with
a few functions defined give a standardized interface for the execution and 
evaluation code. The our tasks can be very different from each other, e.g.,:

- `tasks/navigate.py`: The raw data are choice questions with possible answers 
included in an `input` value to the key `question`. The answer is the value
to the key `target`.
```python
    {'input': 'If you follow these instructions, do you return to the starting point? Always face forward. Take 1 step backward. Take 9 steps left. Take 2 steps backward. Take 6 steps forward. Take 4 steps forward. Take 4 steps backward. Take 3 steps right.\nOptions:\n- Yes\n- No', 
    'target': 'No'}
```
- `tasks/professional_accounting`: The raw data are differently structured
    with different keys that convey a multiple-choice question in a varied way. 
```python
    {'question': 'You bought a limousine for $98,000 and are planning to rent it for weddings, ceremonies and parties at $245 per hour. If you estimate the car will be hired for 2 hours a day on average, with daily costs at about $50, what is the estimated yearly yield on your investment if you work all year round, i.e. every day of the year, including any festivities and weekends?', 
    'subject': 'professional_accounting', 
    'choices': ['164%', '1.64%', '0.45%', '183%'], 
    'answer': 0}
```
In order to keep the execution code, `run_experiment.py`, from being a huge 
`if/then/else` mess to massage each type of format into a LLM ready payload, we instead have each task module give a standardized format for processing. 
- The standardized format for our V2 experiments with 0-shot (explained below) is:
```python
        {'input': 'You bought a limousine for $98,000 and are planning to rent it for weddings, ceremonies and parties at $245 per hour. If you estimate the car will be hired for 2 hours a day on average, with daily costs at about $50, what is the estimated yearly yield on your investment if you work all year round, i.e. every day of the year, including any festivities and weekends?\n(A) 164%. (B) 1.64%. (C) 0.45%. (D) 183%. ', 
        'target': '(A)'}
```
- The standardized format reformats multiple choices into A-D options and 
converts the answer to one of the A-D options. The conversion to A-D was for 
ease of parsing out the answer. The `navigate` option is left as is since `Yes` and `No` are easy to parse.  

- There are additional task variations that have to be accounted for. There are
both 0-shot and few-shot versions in the v2 paper. In `tests/test_tasks` there
are examples of how to invoke the two versions and the difference in the 
`few-shot` example are shown. A confusing detail is that 'few-shot' has 
different notions of 'few'--see the v2 paper but for consistency we stick with
the label. 

### Evaluation: 

The tasks have different needs getting an answer and scoring. There are 
several ways needed to find the answer to the multiple choice questions and 
this becomes even more challenging if the LLM response is open ended like 
programming challenges. The evaluation metrics are:

- TARr@N: Total agreement rate raw @ N runs: This is the simplest metric and generally will be the same independent of configuration. Typical implementation 
is: 

```python
    def raw_fn(row: pd.Series)-> str: 
        """
        Returns raw LLM response. 
        Arguments:
            row: Has raw LLM response in 'response' index
        Returns:
            str
        """
        return row['response']
```

- TARa@N: Total agreement rate answer @ N runs: The tasks should try and 
defer to functions in `helper_functions.py` because LLMs can be quite 
creative and the UPBFS (uniquely presupposed breadth first search) works 
well for that task. Look in 
`tests/test_helper_functions.py::test_parse_parenthesized_answers` and
`tests/test_helper_functions.py::test_parse_string` for examples of 
how the answer parsing works. The goal is to abstract from semantically equivalent answers that are not string equivalent:
    + '(A)' recognized: 
        - "The answer is A"
        - "A is the answer"
        - "(A)"
        - "(a)"
        - "The answer is (A), meaning (A)"
    + '(B)' recognized because matching has not generalized to bare 'A':
        -  "A good answer is (B)"--ignores 'A'
    + Blown uniqueness presupposition means more than one answer found:
        - "Yes, but No" -> Blown uniqueness presupposition (UP)
        -  "A good answer is B" -> Blown UP because 'A' and 'B' are matched
    + 'Yes' not recognized:
        - "Yesterday"-> None

Look at the implementation and tests to see how this all works. 

The code is in `answer_fn` and for v2 data it routes 
to the appropriate answer set based on the configuration. 

```python
def answer_fn(row, config): 
    """
    Returns parsed answer from LLM response based on config. 

    Args:
        row (pd.Series): Has raw LLM response in 'response' index
        config (dict): Configuration for the task
    Returns:
        str: Answer in original format or None if no answer found
    Raises:
        LookupError for Blown Uniqueness Presupposition
    """
    if config['prompt_type'] == "v2":
        answers = ['Yes', 'No']
        return helper_functions.parse_string(answers, row)
```

- Correct: Does the parsed system response match the truth? This function
reduces the evaluation of the LLM and task to Python equivalence, for v2 
it is str equivalence between the `row['gt']` value and parsed answer produced above, `True` or `False`, with two additional values, an exception `LookupError` that corresponds to when there is more than one answer found and `None` if no answer is found. It is up to the calling process of `correct_fn` to decide what to do with these 4 values. An example from `navigate.py` is:

```python
def correct_fn(row: pd.Series, config:dict)-> (bool, None):
    """Determines if system response is correct with Python string
    equivalence. Implements 4 value response, (True, False, None, Blown UP). 
    Args:
        row (pd.Series): Row from task
        config (dict): Configuration for task
    Returns:
        bool: Whether answer is correct, None if no answer
    Raises:
        LookupError for Blown Uniqueness Presupposition
    """
    answer = answer_fn(row, config) #throws LookupError Blown UP
    if answer is None:
        return answer
    return row['gt'] == answer
```






