import base64
import requests
import sys
import os
import os.path
import time

from openai import OpenAI
api_key = os.environ["OPENAI_API_KEY"]

DELIMITER = "*****"
QA_DELIMITER = "@@@@@@@@@@"

def create_image_payload(api_key, image_path, prompt, max_tokens=300):
    if not os.path.isfile(image_path):
        return None, None
    
    with open(image_path, "rb") as fp:
        base64_image = base64.b64encode(fp.read()).decode('utf-8')

    headers = {
      "Content-Type": "application/json",
      "Authorization": f"Bearer {api_key}"
    }
    
    image_url = {
        "url": f"data:image/jpeg;base64,{base64_image}",
        "detail": "low"
    }
    
    payload = {
      "model": "gpt-4-vision-preview",
      "messages": [
        {
          "role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": image_url}
          ]
        }
      ],
      "max_tokens": 300
    }
    return headers, payload

def send_gpt4v_request(headers, payload, debug=False, max_requests=6):
    # list of responses and last submitted payload
    response_list = []
    
    for i in range(max_requests): # limiting number of continue requests
        if debug:
            print("[pushing request]", i+1)
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        try:
            if "error" in list(response.json().keys()): # server error
                if debug:
                    print("[server error]")
                continue
        except:
            print("ERROR, waiting 5 sec")
            print(response)
            time.sleep(5)
            continue
        response_list.append(response.json())
    
        finish_reason = response_list[-1]["choices"][0]["finish_reason"]
        response_msg = response_list[-1]["choices"][0]["message"]
        if debug:
            print(response_msg["content"])

        if finish_reason != "length":
            break
    
        payload["messages"].append(response_msg)
        payload["messages"].append({'role': 'user', 'content': 'continue'})
    
        if debug:
            print("[continuing...]")
    if i == max_requests-1: # probably haven't finished request
        return None, None
    return response_list, payload

def extract_text_from_response_list(response_list):
    return " ".join([resp["choices"][0]["message"]["content"] for resp in response_list])


def main(rows, output_path):
    server_error_count = 0
    for row_ in tqdm(rows):
        if server_error_count > 30:
            print(f"ERROR: Too many cumulative errors, exiting prematurely.")
            return

        row = row_.strip().split("|")

        row_id = row[0]
        image_path = f"charades_frames/{row[0]}.jpg"
        prompt = f"""This is one frame from a longer video clip where people have described as "{row[2]}" The particular action or activity this frame was selected from is "{row[1]}" Given this context, describe in detail what actions are happening in the frame."""

        headers, payload = create_image_payload(api_key, image_path, prompt, max_tokens=300)
        if headers is None or payload is None:
            print(f"ERROR: {image_path} image does not exist. skipping.")
            continue
        response_list, last_payload = send_gpt4v_request(headers, payload, debug=False)
        if response_list is None:
            print(f"ERROR: {image_path} inadequate response for description. skipping.")
            server_error_count += 1
            continue
        desc_msg = extract_text_from_response_list(response_list)

        last_payload["messages"].append({
            'role': 'user',
            'content': "List out at questions and corresponding detailed answers in bullet points regarding what I can ask about the image, specifically related to activities, actions, movement, and/or temporal relationships."
        })
        last_payload["max_tokens"] = 600
        qa_response_list, last_payload = send_gpt4v_request(headers, last_payload, debug=False)
        if qa_response_list is None:
            print(f"ERROR: {image_path} inadequate response for qa. skipping.")
            server_error_count += 1
            continue
        qa_msg = extract_text_from_response_list(qa_response_list)

        with open(output_path, "a+") as fp:
            fp.write(f"{DELIMITER} {row_id} {DELIMITER}\n")
            fp.write(desc_msg + "\n")
            fp.write(f"{QA_DELIMITER}\n")
            fp.write(qa_msg + "\n")

from tqdm import tqdm
if __name__ == '__main__':
    frame_file = "data_head.txt"
    output_file = "gpt4_query_output.txt"
    with open(frame_file, "r") as fp:
        rows = fp.readlines()

    data = rows
    print(data[0].strip())
    main(data, output_file)
