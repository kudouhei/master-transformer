# MMLU
# MMLU seeks to test whether a model understands language well enough to answer second-tier questions about subjects such as history, mathematics, morality, and law.
#

from deepeval.benchmarks import MMLU
from deepeval.benchmarks.tasks import MMLUTask
from deepeval.models.base_model import DeepEvalBaseLLM
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# set up the model
class DeepEvalModel(DeepEvalBaseLLM):
    def __init__(self, model, tokenizer, name):
        self.model = model
        self.tokenizer = tokenizer
        self.name = name

        device = torch.device("cube" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.device = device

    def load_model(self):
        return self.model
    
    def generate(self, prompt: str) -> str:
        model = self.load_model()
        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(
            self.device
        )

        generated_ids = model.generate(
            **model_inputs, max_new_tokens=100, do_sample=True
        )
        return self.tokenizer.batch_decode(generated_ids)[0]

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return self.name

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

gpt2 = DeepEvalModel(model=model, tokenizer=tokenizer, name="GPT-2")

# Define benchmark with specific tasks and shots
benchmark = MMLU(
    tasks=[MMLUTask.HIGH_SCHOOL_COMPUTER_SCIENCE, MMLUTask.ASTRONOMY],
    n_shots=3,
)

# Run benchmark
benchmark.evaluate(model=gpt2)
print(benchmark.overall_score)