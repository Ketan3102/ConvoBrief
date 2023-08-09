import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,pipeline
from peft import PeftModel, PeftConfig
import streamlit as st

@st.cache_resource
def load_model():
    config = PeftConfig.from_pretrained("Ketan3101/ConvoBrief")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
    model = PeftModel.from_pretrained(model, "Ketan3101/ConvoBrief")
    tokenizer=AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    return model, tokenizer

def main():
    st.set_page_config(page_title="ConvoBrief", page_icon="📝")
    model,tokenizer=load_model()
    st.title("ConvoBrief: A dialogue summarizer")
    dialogue=st.text_area("Enter the Dialogue")

    if st.button("Summarize Dialogue"):
        if dialogue:
            inputs=tokenizer(dialogue,return_tensors='pt')
            summary=tokenizer.decode(
                model.generate(input_ids=inputs['input_ids'], max_new_tokens=200, temperature=1.2001, do_sample=True)[0],
                skip_special_tokens=True
            )

            st.subheader("Summarized Dialogue:")
            st.write(summary)
            st.error("The model has been trained on less parameters, so their might be minor errors")

        else:
            st.warning("No! Dialogue was given")

if __name__=="__main__":
    main()