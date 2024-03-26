#%%
import openai 
import os
# %%
openai.api_key = os.getenv("OPENAIAPI")
# %%
response = openai.Image.create(
    prompt = "a black dog with a tiny hat",
    n=1,
    size="1024x1024"
)
image_url = response['data'][0]['url']
# %% Editing
