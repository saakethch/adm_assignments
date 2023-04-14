import streamlit as st
import pinecone
import openai

st.title("ADM Assignment 5 - ")
index_name = 'openai-youtube-mac-transcriptions-a5'

# initialize connection to pinecone (get API key at app.pinecone.io)
pinecone.init(
    api_key="cd5700c8-f94e-4688-94ee-20c4c451d36b",
    environment="northamerica-northeast1-gcp"
)

# check if index already exists (it shouldn't if this is first time)
if index_name not in pinecone.list_indexes():
    # if does not exist, create index
    pinecone.create_index(
        index_name,
        dimension=len(res['data'][0]['embedding']),
        metric='cosine',
        metadata_config={'indexed': ['channel_id', 'published']}
    )

# connect to index
index = pinecone.Index(index_name)

limit = 3750
openai.api_key = st.secrets["OPENAI_API_KEY"]
def retrieve(query):
    res = openai.Embedding.create(
        input=[query],
        engine="text-embedding-ada-002"
    )

    # retrieve from Pinecone
    xq = res['data'][0]['embedding']

    # get relevant contexts
    res = index.query(xq, top_k=3, include_metadata=True)
    contexts = [
        x['metadata']['text'] for x in res['matches']
    ]

    # build our prompt with the retrieved contexts included
    prompt_start = (
        "Answer the question based on the context below.\n\n"+
        "Context:\n"
    )
    prompt_end = (
        f"\n\nQuestion: {query}\nAnswer:"
    )
    # append contexts until hitting limit
    for i in range(1, len(contexts)):
        if len("\n\n---\n\n".join(contexts[:i])) >= limit:
            prompt = (
                prompt_start +
                "\n\n---\n\n".join(contexts[:i-1]) +
                prompt_end
            )
            break
        elif i == len(contexts)-1:
            prompt = (
                prompt_start +
                "\n\n---\n\n".join(contexts) +
                prompt_end
            )
    return prompt

def complete(prompt):
    # query text-davinci-003
    res = openai.Completion.create(
        engine='text-davinci-002',
        prompt=prompt,
        temperature=0,
        max_tokens=400,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    return res['choices'][0]['text'].strip()

def ask_me(prompt):
    query = prompt
    query_with_contexts = retrieve(query)
    return complete(query_with_contexts)

# user input
apparel_prompt = st.text_input("Enter a question related to Canon EOS R5 : ")
ans = ask_me(apparel_prompt)
st.write(ans)
