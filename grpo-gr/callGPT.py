from openai import AzureOpenAI

from gpt_credentials import api_base, api_key, deployment_name, api_version

client = AzureOpenAI(
    api_key=api_key,  
    api_version=api_version,
    base_url=f"{api_base}/openai/deployments/{deployment_name}"
)



import base64, json
from mimetypes import guess_type



# Function to encode a local image into data URL 
def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"

def call_gpt(prompt, image_path_0 = None, image_path_1 = None, image_path_2 = None, image_path_3 = None,):
    # Path to your image
    # image_path = "/data3/yue/DocVQA_task1/images/ffbf0023_4.png"

    ms = [
                # { "role": "system", "content": "You are a helpful assistant." },

                { "role": "user", 
                 "content": [  
                        { 
                            "type": "text", 
                            "text": prompt
                        },
                        
                        
                    ] 
                } 
            ]
    if image_path_0:
        ms[0]['content'].append({ 
                            "type": "image_url",
                            "image_url": {
                            "url": local_image_to_data_url(image_path_0)
                            }
                        })
    if image_path_1:
        ms[0]['content'].append({ 
                            "type": "image_url",
                            "image_url": {
                            "url": local_image_to_data_url(image_path_1)
                            }
                        })
    if image_path_2:
        ms[0]['content'].append({ 
                            "type": "image_url",
                            "image_url": {
                            "url": local_image_to_data_url(image_path_2)
                            }
                        })
    if image_path_3:
        ms[0]['content'].append({ 
                            "type": "image_url",
                            "image_url": {
                            "url": local_image_to_data_url(image_path_3)
                            }
                        })

    
    try:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=ms         )
        response = json.loads(response.json() )
        return response['choices'][0]['message']['content']
    except Exception as e:
        # print(response)
        print(e)
        return 'I do not know.'