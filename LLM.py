import openai
import tiktoken
import time
enc = tiktoken.get_encoding("cl100k_base")
assert enc.decode(enc.encode("hello world")) == "hello world"
enc = tiktoken.encoding_for_model("gpt-4")

openai_api_key_name = 'sk-...'

def GPT_response(messages, model_name):
  token_num_count = 0
  for item in messages:
    token_num_count += len(enc.encode(item["content"]))

  if model_name in ['gpt-4', 'gpt-4-32k', 'gpt-3.5-turbo-0301', 'gpt-4-0613', 'gpt-4-32k-0613', 'gpt-3.5-turbo-16k-0613']:
    #print(f'-------------------Model name: {model_name}-------------------')
    openai.api_key = openai_api_key_name

    try:
      result = openai.ChatCompletion.create(
        model=model_name,
        messages=messages,
        temperature = 0.0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
      )
    except:
      try:
        result = openai.ChatCompletion.create(
          model=model_name,
          messages=messages,
          temperature=0.0,
          top_p=1,
          frequency_penalty=0,
          presence_penalty=0
        )
      except:
        try:
          print(f'{model_name} Waiting 60 seconds for API query')
          time.sleep(60)
          result = openai.ChatCompletion.create(
            model=model_name,
            messages=messages,
            temperature = 0.0,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
          )
        except:
          return 'Out of tokens', token_num_count
    token_num_count += len(enc.encode(result.choices[0]['message']['content']))
    print(f'Token_num_count: {token_num_count}')
    return result.choices[0]['message']['content'], token_num_count

  else:
    raise ValueError(f'Invalid model name: {model_name}')
