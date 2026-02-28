from hf_tokenizer import HfBPETokenizerVisualizer

visualizer = HfBPETokenizerVisualizer(
    pretrained_model_name="gpt2",
    save_visualization=True,
    file_type="png",
    file_name="bpe_tokenization_visualization",
    enable_debug=True,
)

encoded_ids = visualizer.encode("hello world")
print(encoded_ids)
