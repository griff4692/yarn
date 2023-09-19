from scaled_rope.configuration_llama import LlamaConfig


OUT_DIR = '/nlp/projects/summarization/bhc_data_cleanup/bhc_weights/yarn-7b-8k-none/step_2000'
MODEL = 'NousResearch/Llama-2-7b-hf'
YARN_FACTOR = 2
MAX_ORIGINAL = 4096


if __name__ == '__main__':
    config = LlamaConfig.from_pretrained(MODEL)
    config.rope_scaling = {
        "type": "yarn",
        "factor": YARN_FACTOR,
        "original_max_position_embeddings": MAX_ORIGINAL
    }

    config.max_position_embeddings = int(YARN_FACTOR * MAX_ORIGINAL)
    config.save_pretrained(OUT_DIR)
