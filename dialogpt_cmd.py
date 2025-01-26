from transformers import AutoTokenizer, AutoModelForCausalLM
import torch._dynamo
torch._dynamo.config.suppress_errors = True

tokenizer = AutoTokenizer.from_pretrained("pineappleSoup/DialoGPT-medium-707")
model = AutoModelForCausalLM.from_pretrained("pineappleSoup/DialoGPT-medium-707")

def interact_with_model(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        max_length=200,
        num_return_sequences=1
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Start the loop
if __name__ == "__main__":
    print("Start chatting with the model! Type 'exit!' to quit.\n")
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() == "exit!":
            print("Goodbye!")
            break
        response = interact_with_model(user_input)
        print("Model:", response)