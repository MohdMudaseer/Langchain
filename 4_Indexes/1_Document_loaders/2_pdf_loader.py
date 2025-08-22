from langchain_community.document_loaders import PyMuPDFLoader
# from langchain_community.document_loaders import PyPDFLoader
loader = PyMuPDFLoader(r'4_Indexes\1_Document_loaders\dl-curriculum.pdf')
# loader=PyPDFLoader(r'4_Indexes\1_Document_loaders\dl-curriculum.pdf')


docs = loader.load()

print(len(docs))

print(docs[0].page_content)
print(docs[1].metadata)
