# HuggingFace Byte-Pair Encoding tokenizer visualizer library

The library can help you visualize how the encoding process happens in the **Byte-Pair Encoding** tokenizer algorithm when you pass on your text content for tokenization.

Byte-Pair Encoding (BPE) was initially developed as an algorithm to compress texts, and then used by OpenAI for tokenization when pre-training the GPT model. It’s used by a lot of Transformer models, including GPT, GPT-2, RoBERTa, BART, and DeBERTa.

![visualizer-front](./media/front-1.png)

## Byte-Pair Encoding tokenization

BPE training starts by computing the unique set of words used in the corpus (after the normalization and pre-tokenization steps are completed), then building the vocabulary by taking all the symbols used to write those words.

More about the algorithm [here](https://huggingface.co/learn/llm-course/en/chapter6/5)

![tiktoken-img](https://raw.githubusercontent.com/SauravP97/hf-tokenizer-visualizer/refs/heads/main/media/img1.png)

## Visualizing the Tokenization process

During the tokenization process the input content is compressed into the encoded IDs based on the trained BPE-Tokenizer. During the training process the token-pairs are merged into new token ID based on their frequency of existence in the training corpus.

This library helps in visualizing how the merging process looks like for a given string to be encoded. It generates a **graph** where the nodes are tokens / characters and if a pair of characters are merged, the nodes are connected via directed edges.

### Installing the Library

```python
pip install hf-tokenizer-visualizer
```
### Using the Library

1. Save your visualization in a `PNG` or a `PDF` file.

```python
from hf_tokenizer_visualizer import HfBPETokenizerVisualizer

visualizer = HfBPETokenizerVisualizer(
    pretrained_model_name="gpt2",
    save_visualization=True,
    file_type="png",
    file_name="bpe_tokenization_visualization_2",
    enable_debug=False,
)

visualizer.visualize_encoding('hello world')
```

> Note: The file is saved in your current working directory.
> Note: You can choose between `png` and `pdf` file types.

2. Get the raw encoding

```python
from hf_tokenizer_visualizer import HfBPETokenizerVisualizer

visualizer = HfBPETokenizerVisualizer(
    pretrained_model_name="gpt2",
    save_visualization=True,
    file_type="png",
    file_name="bpe_tokenization_visualization_2",
    enable_debug=False,
)

visualizer.encode('hello world')
```

> Output: [31373, 995]

### Output Graph generated

![generated graph](https://raw.githubusercontent.com/SauravP97/hf-tokenizer-visualizer/refs/heads/main/media/generated_graph.png)

#### Building the Library and Uploading to PyPi (Owner Only)

```
python -m build
twine upload --repository pypi dist/*
```

### Pre-requisites

Following are the libraries which are mandatory for running this:
  - [graphviz](https://pypi.org/project/graphviz/)
  - [transformers](https://pypi.org/project/transformers/)