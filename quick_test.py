from mlx_lm import load, generate

from mlx_lm.processor_utils import process_vision_info


model, tokenizer, processor = load("mlx-community/Qwen2.5-VL-3B-Instruct-4bit")

prompt = "Describe the image"

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "http://images.cocodataset.org/val2017/000000039769.jpg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
print("PROMPT: ", prompt)
images = process_vision_info(messages)
print("Images: ", images)

generate(model, tokenizer, prompt=prompt, processor=processor, images=images, prefill_step_size=4096, verbose=True)


# from mlx_lm import load, generate

# model, tokenizer = load("mlx-community/Mistral-7B-Instruct-v0.3-4bit")

# prompt = "Write a story about Einstein"

# messages = [{"role": "user", "content": prompt}]
# prompt = tokenizer.apply_chat_template(
#     messages, add_generation_prompt=True
# )

# generate(model, tokenizer, prompt=prompt, verbose=True)
