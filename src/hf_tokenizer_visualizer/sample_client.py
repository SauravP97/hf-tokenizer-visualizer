from hf_tokenizer import HfBPETokenizerVisualizer

visualizer = HfBPETokenizerVisualizer(
    pretrained_model_name="gpt2",
    save_visualization=True,
    file_type="png",
    file_name="bpe_tokenization_visualization",
    enable_debug=True,
)

visualizer.visualize_encoding("The library can help you visualize how the")
