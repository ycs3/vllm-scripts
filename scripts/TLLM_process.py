import pickle
import time
from openai import OpenAI
import os

DELIMITER = "###"

if __name__ == '__main__':
    model_type = "GPT4V" # vision LLM type
    finetune_type = "none" # finetune type
    subset_size = 128

    use_openai = True
    tllm_model_type = "GPT4" # text LLM type
    api_model = "gpt-4-turbo-preview"

    #use_openai = False
    #tllm_model_type = "Qwen7B" # text LLM type
    #api_model = "Qwen/Qwen1.5-7B-Chat"
    #tllm_model_type = "Qwen14B" # text LLM type
    #api_model = "Qwen/Qwen1.5-14B-Chat"
    #tllm_model_type = "Gemma7B" # text LLM type
    #api_model = "google/gemma-7b-it"

    if use_openai is True:
        API_KEY = os.environ["OPENAI_API_KEY"]
    else:
        API_KEY = os.environ["TOGETHER_API_KEY"]

    with open(f"vllm_db_{model_type}_{finetune_type}_{subset_size}.pkl", "rb") as fp:
        pairs, query_db = pickle.load(fp)

    if use_openai is True:
        client = OpenAI(api_key=API_KEY)
    else:
        client = OpenAI(
            api_key=API_KEY,
            base_url="https://api.together.xyz/v1"
        )
    for a1, b2, action_query, a2, a3 in pairs:
        if not use_openai:
            time.sleep(.9) # together API rate limit
        print(a1, action_query)
        frame_description_list = query_db[a1][b2]
        frame_descriptions = "\n".join([f" * Frame {idx+1}: {val}" for idx, val in enumerate(frame_description_list)])
        text_llm_query = "The following are descriptions of actions for frames extracted 1 second apart from a video clip:\n\n"+\
        frame_descriptions + "\n\n" +\
        f"The action '{action_query}' has occurred in the video clip. What interval is the action most likely to start and end? Provide your best guess by providing the start and end frame numbers in json format."
        chat_completion = client.chat.completions.create(
            messages = [
                {
                    "role": "user",
                    "content": text_llm_query,
                }
            ],
            model=api_model
        )
        text_response = chat_completion.choices[0].message.content

        with open(f"tllm_db_{model_type}_{finetune_type}_{tllm_model_type}_{subset_size}.txt", "a+") as fp:
            fp.write(DELIMITER + a1 + DELIMITER + action_query + "\n")
            fp.write(text_response + "\n")
