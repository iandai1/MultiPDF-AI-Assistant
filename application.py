import streamlit as st
import pdfplumber

def get_pdf_text(pdf_documents):
    text = ""
    for pdf in pdf_documents:
        with pdfplumber.open(pdf) as pdf_reader:
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
    return text
                

def main():
    st.set_page_config(page_title="Ask question about your pdfs")
    st.header("PDF Document Assistant")
    st.chat_input("Ask question about your pdf...")
    with st.sidebar:
        st.subheader("Your documents")
        pdf_documents = st.file_uploader(
            "Upload your documents and click on 'Process'",  accept_multiple_files=True
        )
        if st.button("Process"):
            if not pdf_documents:
                st.error("Upload at least one doc")
                return
            
            with st.spinner("Processing"):
                #get pdf text
                raw_text = get_pdf_text(pdf_documents)
                st.write(raw_text)
                
                #get the text chunks
                
                #create vector store
                
                #create conversation chain
                st.success(f"Documents {len(pdf_documents)} processed successfully")
    
    

if __name__ == '__main__':
    main()