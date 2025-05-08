## Prompt Engineering

A prompt is an instruction given to a model to perform a task.

A prompt generally consists of one or more of the following parts:
- Task description
  - What you want the model to do, including the role you want the model to play and the output format.
- Example(s) of how to do this task
  - For example, if you want the model to detect toxicity in text, you might provide a few examples of what toxicity and non-toxicity look like.
- The task
  - The concrete task you want the model to do, such as the question to answer or the book to summarize.
  
`How much prompt engineering is needed depends on how robust the model is to prompt perturbation.`

**System Prompt and User Prompt**
Many model APIs give you the option to split a prompt into a `system prompt` and a `user prompt`.

- The `system prompt` is a prompt that sets the behavior of the model.
- The `user prompt` is a prompt that contains the actual task you want the model to do.

**Write Clear and Explicit Instructions**
- Explain, without ambiguity, what you want the model to do
- Ask the model to adopt a persona
- Provide examples
- Specify the output format