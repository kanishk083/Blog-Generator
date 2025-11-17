import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers

##function to get response from llma-2 

def get_llm_response(input_text,no_words,blog_style):
    #llm
    model_path = r"C:\Users\KANISHK\OneDrive\Desktop\Blog Generation\models\llama-2-7b-chat.ggmlv3.q4_K_S.bin"
    llm=CTransformers(model=model_path,
                      model_type='llama',
                      config={'max_new_tokens':256,
                              'temperature':0.01}  )
    #prompt template

    template = """
    Write a blog for {blog_style} on the topic "{input_text}" within {no_words} words.
    """

    prompt = PromptTemplate(
        input_variables=['blog_style', 'input_text', 'no_words'],
        template=template
    )

    final_prompt = prompt.format(
        blog_style=blog_style,
        input_text=input_text,
        no_words=no_words,
    )

    
    #get response

    response = llm.invoke(final_prompt)
    print(response)
    return response







st.set_page_config(page_title="Blog Post Generator",
                   page_icon="📝",
                   layout="centered",
                   initial_sidebar_state="collapsed")
st.header("Generate Blog 📝")

input_text = st.text_input("Enter the topic")


##two cols for additional options
col1,col2=st.columns([5,5]) #width ratio 5:5

with col1:
    no_words = st.text_input("No of words")
with col2:
    blog_style = st.selectbox("Writing the blog for",
                              ('Researchers','Data-Scientists','AI-Engineers','Students','Hobbyists','General Public'))
    
submit = st.button("Generate Blog")

#dsiplay output
if submit:
    st.write(get_llm_response(input_text,no_words,blog_style))