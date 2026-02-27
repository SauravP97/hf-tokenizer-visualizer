from transformers import AutoTokenizer
from tokenizers.models import BPE
from tokenizers import Tokenizer

import json
import graphviz


class HfBPETokenizerVisualizer:
    """A class to visualize the BPE tokenization process using a Hugging Face tokenizer."""

    BPE_TOKENIZER_ASSERTION_ERROR_MESSAGE = (
        "The tokenizer's model is not a Byte-Pair Encoding model."
    )
    GRAPHVIZ_RENDERING_ENGINE = "neato"

    def __init__(
        self,
        *,
        pretrained_model_name: str,
        save_visualization: bool = False,
        file_type: str = "png",
        file_name: str = "visualized_graph",
        enable_debug: bool = False,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.bpe_class: BPE = self.tokenizer.backend_tokenizer.model
        assert isinstance(
            self.bpe_class, BPE
        ), self.BPE_TOKENIZER_ASSERTION_ERROR_MESSAGE

        bpe_tokenizer = Tokenizer(self.bpe_class)
        tokenizer_json = bpe_tokenizer.to_str()
        tokenizer_config = json.loads(tokenizer_json)

        self.vocab: dict = tokenizer_config["model"]["vocab"]
        self.merges: list = tokenizer_config["model"]["merges"]
        self.save_visualization = save_visualization
        self.file_type = file_type
        assert self.file_type in {
            "png",
            "pdf",
        }, "Unsupported File Type: file_type must be either 'png' or 'pdf'"

        self.file_name = file_name
        self.enable_debug = enable_debug

    def encode(self, text: str) -> list[int]:
        """Encodes the input text using the BPE tokenizer and returns a list of token IDs."""
        return self.tokenizer.encode(text)

    def visualize_encoding(self, text: str):
        """Visualizes the BPE tokenization process for a given input text."""
        dot = graphviz.Digraph(
            comment="Tokenizer Visualizer", engine=self.GRAPHVIZ_RENDERING_ENGINE
        )
        existing_nodes = set()

        text_chunks: list[str] = []
        for char in text:
            dot.node(char)
            text_chunks.append(char)
            if self.enable_debug:
                print(f"Adding node for character '{char}' with index {ord(char)}")
            existing_nodes.add(char)

        for merged_indices in self.merges:
            assert (
                len(merged_indices) == 2
            ), "Each merge should consist of exactly two indices."
            pair: tuple[str, str] = (merged_indices[0], merged_indices[1])
            text_chunks = self._merge(text_chunks, pair, existing_nodes, dot)

        if self.save_visualization:
            dot.render(self.file_name, format=self.file_type, cleanup=True)
        else:
            print(dot)

    def _merge(
        self,
        text_chunks: list[str],
        pair: tuple[str, str],
        existing_nodes: set[str],
        dot: graphviz.Digraph,
    ) -> list[str]:
        merged_chunks = []
        i = 0
        while i < len(text_chunks):
            if (
                i < (len(text_chunks) - 1)
                and text_chunks[i] == pair[0]
                and text_chunks[i + 1] == pair[1]
            ):
                merged_char = pair[0] + pair[1]
                merged_chunks.append(merged_char)
                if merged_char not in existing_nodes:
                    if self.enable_debug:
                        print(
                            f"Merging '{pair[0]}' and '{pair[1]}' to form '{merged_char}'"
                        )
                    existing_nodes.add(merged_char)
                dot.node(merged_char)
                dot.edge(pair[0], merged_char, label="")
                dot.edge(pair[1], merged_char, label="")
                i += 2
            else:
                merged_chunks.append(text_chunks[i])
                i += 1
        return merged_chunks
