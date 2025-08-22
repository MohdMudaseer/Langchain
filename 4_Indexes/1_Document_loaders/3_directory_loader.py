from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader

loader = DirectoryLoader(
    path='sample_path',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

docs = loader.lazy_load()

for document in docs:
    print(document.metadata)

# *.pdf= all pdf files
# **/*.txt= all text files in all subdirectories
# data/*.csv= all csv files in the data subdirectory
# **/*= all files in all subdirectories