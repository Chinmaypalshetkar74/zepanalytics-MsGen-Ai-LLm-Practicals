.. code:: ipython3

    # Importing Libraries
    from langchain_community.llms import Ollama
    from langchain_ollama import OllamaLLM

-  from langchain_community.llms import Ollama: Imports the community
   version of Ollama LLM wrapper for LangChain. May be used in older
   setups or custom workflows.
-  from langchain_ollama import OllamaLLM: Imports the official Ollama
   integration for LangChain. Preferred for current, stable use with
   local models like LLaMA, Mistral, etc. Use one of them depending on
   your setup. For new projects, go with OllamaLLM.

.. code:: ipython3

    
    # Initialize the model
    llm = Ollama(model='llama3.1:8b')


.. parsed-literal::

    C:\Users\weare\AppData\Local\Temp\ipykernel_16784\2378445012.py:2: LangChainDeprecationWarning: The class `Ollama` was deprecated in LangChain 0.3.1 and will be removed in 1.0.0. An updated version of the class exists in the `langchain-ollama package and should be used instead. To use it run `pip install -U `langchain-ollama` and import as `from `langchain_ollama import OllamaLLM``.
      llm = Ollama(model='llama3.1:8b')
    

-  llm = Ollama(model=‘llama3.1:8b’): This initializes a local language
   model using Ollama. It loads the model named ‘llama3.1:8b’, which
   likely refers to LLaMA 3.1 with 8 billion parameters. ✅ After this,
   llm can be used to generate text or interact with the model in
   LangChain workflows

.. code:: ipython3

    question = "Popular actor of india"
    response = llm.invoke(question)
    print(response)


.. parsed-literal::

    Here are some of the most popular actors in India:
    
    **Male Actors:**
    
    1. **Shah Rukh Khan**: Known as the "King of Bollywood", he has acted in numerous hit films like Dilwale Dulhania Le Jayenge, Kuch Kuch Hota Hai, and Kabhi Khushi Kabhie Gham.
    2. **Amitabh Bachchan**: A legendary actor who has been active in the industry for over 50 years, known for his iconic roles in films like Sholay, Deewar, and Black.
    3. **Salman Khan**: One of the highest-paid actors in India, known for his blockbuster hits like Bajrangi Bhaijaan, Sultan, and Dabangg.
    4. **Hrithik Roshan**: A versatile actor known for his energetic performances in films like Kaho Naa Pyaar Hai, Dhoom 2, and Kaabil.
    5. **Ranveer Singh**: Known for his energetic and flamboyant performances in films like Bajirao Mastani, Padmaavat, and Gully Boy.
    
    **Female Actors:**
    
    1. **Priyanka Chopra**: A popular actress who has acted in numerous hit films like Kaminey, Barfi!, and Mary Kom.
    2. **Deepika Padukone**: One of the highest-paid actresses in India, known for her performances in films like Om Shanti Om, Cocktail, and Piku.
    3. **Alia Bhatt**: A talented young actress who has quickly risen to fame with hits like Highway, 2 States, and Gully Boy.
    4. **Kareena Kapoor Khan**: Known for her iconic roles in films like Jab We Met, 3 Idiots, and Veere Di Wedding.
    5. **Katrina Kaif**: A popular actress known for her performances in films like Bharat, Ek Tha Tiger, and Zero.
    
    These are just a few examples of the many talented actors working in India's film industry today!
    

.. code:: ipython3

    # Generate answers to a question
    question = "Who is sunil shetty "
    response = llm.invoke(question)
    print(response)


.. parsed-literal::

    Suniel Shetty is an Indian actor, film producer, and television personality who has been active in the Hindi film industry since the late 1980s. He was born on August 19, 1961, in Mulki, Karnataka, India.
    
    Shetty began his acting career with a small role in the 1988 film "Balwan", but it was his breakthrough role as a villain in the 1992 film "Dil" that brought him to prominence. He then went on to play lead roles in several successful films, including "Hum Hain Khalnayak" (1994), "Gopi Kishan" (1994), and "Aaditya" (1995).
    
    Shetty's most notable role was perhaps as the villainous Rakka in Rajiv Mehra's 1993 film "Dilwale Dulhania Le Jayenge", which is one of the highest-grossing films of all time in Indian cinema.
    
    In addition to his acting career, Shetty has also ventured into production with his company, Sunshine Productions. He has produced several films, including "Gadar: Ek Prem Katha" (2001), "Krishna Cottage" (2006), and "Tera Mera Ki Rishta" (2010).
    
    Shetty has also made appearances on television shows, such as the popular dance reality show "Jhalak Dikhhla Jaa" in 2014. He is known for his charismatic personality and has been involved in various social causes, including animal welfare and supporting underprivileged children.
    
    Throughout his career, Shetty has received several awards and nominations, including a Filmfare Award nomination for Best Actor in a Negative Role for "Dilwale Dulhania Le Jayenge".
    

Implementing RAG for custom data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  PyPDFLoader: Loads text content from a PDF file and converts it into
   LangChain Document objects.
-  RecursiveCharacterTextSplitter: Splits large text into smaller chunks
   (e.g., 500–1000 characters) while preserving sentence structure as
   much as possible. Useful for feeding into LLMs that have token
   limits.

.. code:: ipython3

    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import PyPDFLoader


pdf_loader = PyPDFLoader("Attention Is All You Need.pdf")
documents = pdf_loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1100,
    chunk_overlap=140
)
chunks = text_splitter.split_documents(documents)

print("Total Chunks:", len(chunks))
print(chunks[0].page_content[:52])




-  Loads the full PDF into a list of **Document** objects.
-  Each **Document** contains text and metadata (like page number).
-  Breaks the full text into smaller overlapping chunks.
-  **chunk_size=1100**: Each chunk has ~1100 characters.
-  **chunk_overlap=140**: Each chunk overlaps the previous by 140
   characters to preserve context.
-  Prints how many chunks were created.
-  Displays the first 800 characters of the first chunk

Notes -
=======

✅ How to Decide Chunk Size Based on PDF Pages (Correct Method)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  You decide parameters using 3 factors:

-  1️⃣ Step 1 — Understand PDF Type \| PDF Type \| Examples \| Best
   Chunk Size \| \| —————— \| ———————– \| ————— \| \| Research Paper \|
   Transformers, ML papers \| **1500–2000** \| \| Textbooks \| ML, DS
   books \| **1800–2500** \| \| Code/Documentation \| LangChain, APIs \|
   **800–1200** \| \| Stories/Novels \| Fiction, articles \|
   **1000–1500** \| \| Legal/Contracts \| Agreements, policies \|
   **1200–1800** \|

-  2️⃣ Step 2 — Estimate Text Density (Important)

The number of pages alone is meaningless because:

A 50-page paper = ~20,000 words

A 50-page textbook = ~40,000–60,000 words

A 50-page presentation-style PDF = ~4,000 words

So instead, ask:

Does 1 page contain heavy text?

If yes, use larger chunks. - Simple rule: \| Page Text Density \| Signs
\| Chunk Size \| \| —————– \| ————————– \| ————- \| \| High Density \|
Full paragraphs, equations \| **1500–2000** \| \| Medium \| Normal text
\| **1200–1500** \| \| Low \| Bullet slides \| **600–900** \|

-  3️⃣ Step 3 — Simple Formula for Chunk Size Based on Pages

If you still want a formula, here is the best practical one: Formula (ML
research/general books): chunk_size = 35000 / number_of_chunks_you_want

To get good RAG performance:

Best practice: Aim for 50–80 chunks total.

So:

For a 52-page research paper → Target 60–70 chunks

Use:

chunk_size = (total_characters / 60)

But you don’t know characters → so use this shortcut:

Shortcut Rule chunk_size = 1500 + (pages / 10 \* 50)

For 52 pages:

chunk_size = 1500 + (52/10 \* 50) = 1500 + 260 = 1760

Perfect for research papers.

🎯 Final Decision Shortcut (use this always) 📘 If PDF has dense text
(research/math/books):

👉 chunk_size = 1600–2000 👉 chunk_overlap = 200–300

📄 If PDF has normal paragraphs:

👉 chunk_size = 1200–1500 👉 chunk_overlap = 150–200

🖥️ If PDF has slides (PPT-like):

👉 chunk_size = 600–800 👉 chunk_overlap = 100–150

⭐ Super Simple Guide
~~~~~~~~~~~~~~~~~~~~

======= ============== ======================
Pages   Type           Recommended Chunk Size
======= ============== ======================
20–50   Research paper **1500–1800**
50–150  Textbook       **1800–2200**
150–500 Long books     **2000–2500**
10–30   Slides         **600–900**
======= ============== ======================

#### 🔥 Want automatic chunk size selection?

##### here is a Python function:

def auto_chunk_size(pages, density="high"):
    if density == "high":   # Research papers, textbooks
        return min(2000, 1500 + pages * 5)
    elif density == "medium":  # Normal text PDFs
        return min(1800, 1200 + pages * 3)
    else:  # Slides / bullet-style
        return 800


.. code:: ipython3

    
    pdf_loader = PyPDFLoader("Attention Is All You Need.pdf")
    documents = pdf_loader.load()
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1300,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    
    print("Total Chunks:", len(chunks))
    print(chunks[0].page_content[:15])
    
    
    


.. parsed-literal::

    Total Chunks: 39
    Provided proper
    

.. code:: ipython3

    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    
    # Build FAISS vector store
    db = FAISS.from_documents(documents=chunks, embedding=embeddings)
    
    # Create retriever
    retriever = db.as_retriever()
    


.. parsed-literal::

    C:\Users\weare\AppData\Local\Temp\ipykernel_16784\3767687740.py:5: LangChainDeprecationWarning: The class `HuggingFaceEmbeddings` was deprecated in LangChain 0.2.2 and will be removed in 1.0. An updated version of the class exists in the `langchain-huggingface package and should be used instead. To use it run `pip install -U `langchain-huggingface` and import as `from `langchain_huggingface import HuggingFaceEmbeddings``.
      embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    

-  **HuggingFaceEmbeddings**: Loads a sentence transformer model to
   convert text into numerical vectors (embeddings).

-  **FAISS**: A fast vector similarity search library used to store and
   search embeddings efficiently

-  Loads the **all-mpnet-base-v2 model** from Hugging Face.

-  This model turns each text chunk into a dense vector that captures
   its meaning.

-  Converts all **chunks** into vectors using the embedding model.

-  Stores them in a FAISS index for fast similarity search.

-  Converts the **FAISS index** into a retriever object.

-  You can now use **retriever.get_relevant_documents(query)** to fetch
   chunks similar to a user query.

Notes -
=======

✅ Recommended Embedding Models for RAG
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1️⃣ all-mpnet-base-v2 (your current choice)

Model: “sentence-transformers/all-mpnet-base-v2”

Pros:

Very strong semantic understanding

Excellent for short & long texts

High-quality embeddings for English research papers

Cons:

Slightly slower than smaller models

Verdict: ✅ Excellent choice for research papers

2️⃣ Other HuggingFace Sentence Transformers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+-------------+-------------------+------------------------------------+
| Model       | Strengths         | Use Case                           |
+=============+===================+====================================+
| ``all-Min   | Lightweight,      | If you want **faster retrieval**   |
| iLM-L6-v2`` | fast, smaller     | and can sacrifice a little         |
|             | embeddings        | accuracy                           |
+-------------+-------------------+------------------------------------+
| ``all-mpne  | High accuracy     | Best **balance for research        |
| t-base-v2`` |                   | papers**                           |
+-------------+-------------------+------------------------------------+
| ``multi     | Optimized for     | If RAG is mostly **QA-based**      |
| -qa-MiniLM- | **que             |                                    |
| L6-cos-v1`` | stion-answering** |                                    |
+-------------+-------------------+------------------------------------+
| ``all-mpne  | Slightly older    | Slightly less accurate, cheaper    |
| t-base-v1`` | version           |                                    |
+-------------+-------------------+------------------------------------+

3️⃣ For Large Docs / Dense Research Papers

You can use multi-qa-mpnet-base-dot-v1 or all-mpnet-base-v2

Why?

Handles longer contexts

Good for similarity search

Works with FAISS / Chroma / Milvus

4️⃣ Light / Fast Option
HuggingFaceEmbeddings(model_name=“sentence-transformers/all-MiniLM-L6-v2”)

Embedding size: 384 (smaller)

Fast, but slightly less semantic precision

🎯 Recommendation for 15–50 page research papers

Best Accuracy: all-mpnet-base-v2 ✅

Fast + Good Accuracy: all-MiniLM-L6-v2

Your current choice (all-mpnet-base-v2) is perfect for Attention Is All
You Need.

.. code:: ipython3

    llm = OllamaLLM(model="llama3.1:8b",gpu=False)

.. code:: ipython3

    !pip install --upgrade langchain
    


.. parsed-literal::

    Requirement already satisfied: langchain in c:\users\weare\ansel\lib\site-packages (1.1.0)
    Requirement already satisfied: langchain-core<2.0.0,>=1.1.0 in c:\users\weare\ansel\lib\site-packages (from langchain) (1.1.0)
    Requirement already satisfied: langgraph<1.1.0,>=1.0.2 in c:\users\weare\ansel\lib\site-packages (from langchain) (1.0.4)
    Requirement already satisfied: pydantic<3.0.0,>=2.7.4 in c:\users\weare\ansel\lib\site-packages (from langchain) (2.10.3)
    Requirement already satisfied: jsonpatch<2.0.0,>=1.33.0 in c:\users\weare\ansel\lib\site-packages (from langchain-core<2.0.0,>=1.1.0->langchain) (1.33)
    Requirement already satisfied: langsmith<1.0.0,>=0.3.45 in c:\users\weare\ansel\lib\site-packages (from langchain-core<2.0.0,>=1.1.0->langchain) (0.4.49)
    Requirement already satisfied: packaging<26.0.0,>=23.2.0 in c:\users\weare\ansel\lib\site-packages (from langchain-core<2.0.0,>=1.1.0->langchain) (24.2)
    Requirement already satisfied: pyyaml<7.0.0,>=5.3.0 in c:\users\weare\ansel\lib\site-packages (from langchain-core<2.0.0,>=1.1.0->langchain) (6.0.2)
    Requirement already satisfied: tenacity!=8.4.0,<10.0.0,>=8.1.0 in c:\users\weare\ansel\lib\site-packages (from langchain-core<2.0.0,>=1.1.0->langchain) (9.0.0)
    Requirement already satisfied: typing-extensions<5.0.0,>=4.7.0 in c:\users\weare\ansel\lib\site-packages (from langchain-core<2.0.0,>=1.1.0->langchain) (4.12.2)
    Requirement already satisfied: jsonpointer>=1.9 in c:\users\weare\ansel\lib\site-packages (from jsonpatch<2.0.0,>=1.33.0->langchain-core<2.0.0,>=1.1.0->langchain) (2.1)
    Requirement already satisfied: langgraph-checkpoint<4.0.0,>=2.1.0 in c:\users\weare\ansel\lib\site-packages (from langgraph<1.1.0,>=1.0.2->langchain) (3.0.1)
    Requirement already satisfied: langgraph-prebuilt<1.1.0,>=1.0.2 in c:\users\weare\ansel\lib\site-packages (from langgraph<1.1.0,>=1.0.2->langchain) (1.0.5)
    Requirement already satisfied: langgraph-sdk<0.3.0,>=0.2.2 in c:\users\weare\ansel\lib\site-packages (from langgraph<1.1.0,>=1.0.2->langchain) (0.2.10)
    Requirement already satisfied: xxhash>=3.5.0 in c:\users\weare\ansel\lib\site-packages (from langgraph<1.1.0,>=1.0.2->langchain) (3.6.0)
    Requirement already satisfied: ormsgpack>=1.12.0 in c:\users\weare\ansel\lib\site-packages (from langgraph-checkpoint<4.0.0,>=2.1.0->langgraph<1.1.0,>=1.0.2->langchain) (1.12.0)
    Requirement already satisfied: httpx>=0.25.2 in c:\users\weare\ansel\lib\site-packages (from langgraph-sdk<0.3.0,>=0.2.2->langgraph<1.1.0,>=1.0.2->langchain) (0.28.1)
    Requirement already satisfied: orjson>=3.10.1 in c:\users\weare\ansel\lib\site-packages (from langgraph-sdk<0.3.0,>=0.2.2->langgraph<1.1.0,>=1.0.2->langchain) (3.11.3)
    Requirement already satisfied: requests-toolbelt>=1.0.0 in c:\users\weare\ansel\lib\site-packages (from langsmith<1.0.0,>=0.3.45->langchain-core<2.0.0,>=1.1.0->langchain) (1.0.0)
    Requirement already satisfied: requests>=2.0.0 in c:\users\weare\ansel\lib\site-packages (from langsmith<1.0.0,>=0.3.45->langchain-core<2.0.0,>=1.1.0->langchain) (2.32.5)
    Requirement already satisfied: zstandard>=0.23.0 in c:\users\weare\ansel\lib\site-packages (from langsmith<1.0.0,>=0.3.45->langchain-core<2.0.0,>=1.1.0->langchain) (0.23.0)
    Requirement already satisfied: anyio in c:\users\weare\ansel\lib\site-packages (from httpx>=0.25.2->langgraph-sdk<0.3.0,>=0.2.2->langgraph<1.1.0,>=1.0.2->langchain) (4.7.0)
    Requirement already satisfied: certifi in c:\users\weare\ansel\lib\site-packages (from httpx>=0.25.2->langgraph-sdk<0.3.0,>=0.2.2->langgraph<1.1.0,>=1.0.2->langchain) (2025.4.26)
    Requirement already satisfied: httpcore==1.* in c:\users\weare\ansel\lib\site-packages (from httpx>=0.25.2->langgraph-sdk<0.3.0,>=0.2.2->langgraph<1.1.0,>=1.0.2->langchain) (1.0.9)
    Requirement already satisfied: idna in c:\users\weare\ansel\lib\site-packages (from httpx>=0.25.2->langgraph-sdk<0.3.0,>=0.2.2->langgraph<1.1.0,>=1.0.2->langchain) (3.7)
    Requirement already satisfied: h11>=0.16 in c:\users\weare\ansel\lib\site-packages (from httpcore==1.*->httpx>=0.25.2->langgraph-sdk<0.3.0,>=0.2.2->langgraph<1.1.0,>=1.0.2->langchain) (0.16.0)
    Requirement already satisfied: annotated-types>=0.6.0 in c:\users\weare\ansel\lib\site-packages (from pydantic<3.0.0,>=2.7.4->langchain) (0.6.0)
    Requirement already satisfied: pydantic-core==2.27.1 in c:\users\weare\ansel\lib\site-packages (from pydantic<3.0.0,>=2.7.4->langchain) (2.27.1)
    Requirement already satisfied: charset_normalizer<4,>=2 in c:\users\weare\ansel\lib\site-packages (from requests>=2.0.0->langsmith<1.0.0,>=0.3.45->langchain-core<2.0.0,>=1.1.0->langchain) (3.3.2)
    Requirement already satisfied: urllib3<3,>=1.21.1 in c:\users\weare\ansel\lib\site-packages (from requests>=2.0.0->langsmith<1.0.0,>=0.3.45->langchain-core<2.0.0,>=1.1.0->langchain) (2.3.0)
    Requirement already satisfied: sniffio>=1.1 in c:\users\weare\ansel\lib\site-packages (from anyio->httpx>=0.25.2->langgraph-sdk<0.3.0,>=0.2.2->langgraph<1.1.0,>=1.0.2->langchain) (1.3.0)
    

.. code:: ipython3

    import langchain
    print(langchain.__version__)
    


.. parsed-literal::

    1.1.0
    

.. code:: ipython3

    !pip show langchain
    


.. parsed-literal::

    Name: langchain
    Version: 1.1.0
    Summary: Building applications with LLMs through composability
    Home-page: https://docs.langchain.com/
    Author: 
    Author-email: 
    License: MIT
    Location: C:\Users\weare\ansel\Lib\site-packages
    Requires: langchain-core, langgraph, pydantic
    Required-by: 
    

.. code:: ipython3

    !pip install --upgrade langchain
    


.. parsed-literal::

    Requirement already satisfied: langchain in c:\users\weare\ansel\lib\site-packages (1.1.0)
    Requirement already satisfied: langchain-core<2.0.0,>=1.1.0 in c:\users\weare\ansel\lib\site-packages (from langchain) (1.1.0)
    Requirement already satisfied: langgraph<1.1.0,>=1.0.2 in c:\users\weare\ansel\lib\site-packages (from langchain) (1.0.4)
    Requirement already satisfied: pydantic<3.0.0,>=2.7.4 in c:\users\weare\ansel\lib\site-packages (from langchain) (2.10.3)
    Requirement already satisfied: jsonpatch<2.0.0,>=1.33.0 in c:\users\weare\ansel\lib\site-packages (from langchain-core<2.0.0,>=1.1.0->langchain) (1.33)
    Requirement already satisfied: langsmith<1.0.0,>=0.3.45 in c:\users\weare\ansel\lib\site-packages (from langchain-core<2.0.0,>=1.1.0->langchain) (0.4.49)
    Requirement already satisfied: packaging<26.0.0,>=23.2.0 in c:\users\weare\ansel\lib\site-packages (from langchain-core<2.0.0,>=1.1.0->langchain) (24.2)
    Requirement already satisfied: pyyaml<7.0.0,>=5.3.0 in c:\users\weare\ansel\lib\site-packages (from langchain-core<2.0.0,>=1.1.0->langchain) (6.0.2)
    Requirement already satisfied: tenacity!=8.4.0,<10.0.0,>=8.1.0 in c:\users\weare\ansel\lib\site-packages (from langchain-core<2.0.0,>=1.1.0->langchain) (9.0.0)
    Requirement already satisfied: typing-extensions<5.0.0,>=4.7.0 in c:\users\weare\ansel\lib\site-packages (from langchain-core<2.0.0,>=1.1.0->langchain) (4.12.2)
    Requirement already satisfied: jsonpointer>=1.9 in c:\users\weare\ansel\lib\site-packages (from jsonpatch<2.0.0,>=1.33.0->langchain-core<2.0.0,>=1.1.0->langchain) (2.1)
    Requirement already satisfied: langgraph-checkpoint<4.0.0,>=2.1.0 in c:\users\weare\ansel\lib\site-packages (from langgraph<1.1.0,>=1.0.2->langchain) (3.0.1)
    Requirement already satisfied: langgraph-prebuilt<1.1.0,>=1.0.2 in c:\users\weare\ansel\lib\site-packages (from langgraph<1.1.0,>=1.0.2->langchain) (1.0.5)
    Requirement already satisfied: langgraph-sdk<0.3.0,>=0.2.2 in c:\users\weare\ansel\lib\site-packages (from langgraph<1.1.0,>=1.0.2->langchain) (0.2.10)
    Requirement already satisfied: xxhash>=3.5.0 in c:\users\weare\ansel\lib\site-packages (from langgraph<1.1.0,>=1.0.2->langchain) (3.6.0)
    Requirement already satisfied: ormsgpack>=1.12.0 in c:\users\weare\ansel\lib\site-packages (from langgraph-checkpoint<4.0.0,>=2.1.0->langgraph<1.1.0,>=1.0.2->langchain) (1.12.0)
    Requirement already satisfied: httpx>=0.25.2 in c:\users\weare\ansel\lib\site-packages (from langgraph-sdk<0.3.0,>=0.2.2->langgraph<1.1.0,>=1.0.2->langchain) (0.28.1)
    Requirement already satisfied: orjson>=3.10.1 in c:\users\weare\ansel\lib\site-packages (from langgraph-sdk<0.3.0,>=0.2.2->langgraph<1.1.0,>=1.0.2->langchain) (3.11.3)
    Requirement already satisfied: requests-toolbelt>=1.0.0 in c:\users\weare\ansel\lib\site-packages (from langsmith<1.0.0,>=0.3.45->langchain-core<2.0.0,>=1.1.0->langchain) (1.0.0)
    Requirement already satisfied: requests>=2.0.0 in c:\users\weare\ansel\lib\site-packages (from langsmith<1.0.0,>=0.3.45->langchain-core<2.0.0,>=1.1.0->langchain) (2.32.5)
    Requirement already satisfied: zstandard>=0.23.0 in c:\users\weare\ansel\lib\site-packages (from langsmith<1.0.0,>=0.3.45->langchain-core<2.0.0,>=1.1.0->langchain) (0.23.0)
    Requirement already satisfied: anyio in c:\users\weare\ansel\lib\site-packages (from httpx>=0.25.2->langgraph-sdk<0.3.0,>=0.2.2->langgraph<1.1.0,>=1.0.2->langchain) (4.7.0)
    Requirement already satisfied: certifi in c:\users\weare\ansel\lib\site-packages (from httpx>=0.25.2->langgraph-sdk<0.3.0,>=0.2.2->langgraph<1.1.0,>=1.0.2->langchain) (2025.4.26)
    Requirement already satisfied: httpcore==1.* in c:\users\weare\ansel\lib\site-packages (from httpx>=0.25.2->langgraph-sdk<0.3.0,>=0.2.2->langgraph<1.1.0,>=1.0.2->langchain) (1.0.9)
    Requirement already satisfied: idna in c:\users\weare\ansel\lib\site-packages (from httpx>=0.25.2->langgraph-sdk<0.3.0,>=0.2.2->langgraph<1.1.0,>=1.0.2->langchain) (3.7)
    Requirement already satisfied: h11>=0.16 in c:\users\weare\ansel\lib\site-packages (from httpcore==1.*->httpx>=0.25.2->langgraph-sdk<0.3.0,>=0.2.2->langgraph<1.1.0,>=1.0.2->langchain) (0.16.0)
    Requirement already satisfied: annotated-types>=0.6.0 in c:\users\weare\ansel\lib\site-packages (from pydantic<3.0.0,>=2.7.4->langchain) (0.6.0)
    Requirement already satisfied: pydantic-core==2.27.1 in c:\users\weare\ansel\lib\site-packages (from pydantic<3.0.0,>=2.7.4->langchain) (2.27.1)
    Requirement already satisfied: charset_normalizer<4,>=2 in c:\users\weare\ansel\lib\site-packages (from requests>=2.0.0->langsmith<1.0.0,>=0.3.45->langchain-core<2.0.0,>=1.1.0->langchain) (3.3.2)
    Requirement already satisfied: urllib3<3,>=1.21.1 in c:\users\weare\ansel\lib\site-packages (from requests>=2.0.0->langsmith<1.0.0,>=0.3.45->langchain-core<2.0.0,>=1.1.0->langchain) (2.3.0)
    Requirement already satisfied: sniffio>=1.1 in c:\users\weare\ansel\lib\site-packages (from anyio->httpx>=0.25.2->langgraph-sdk<0.3.0,>=0.2.2->langgraph<1.1.0,>=1.0.2->langchain) (1.3.0)
    

.. code:: ipython3

    from langchain_core.prompts import PromptTemplate
    
    prompt = PromptTemplate(
        input_variables=["chat_history", "context", "question"],
        template="""
    You are a helpful AI assistant.
    
    Chat History:
    {chat_history}
    
    Context:
    {context}
    
    Question:
    {question}
    
    Answer:
    """
    )
    
    
    

-  This creates a custom prompt template for a Retrieval-Augmented
   Generation (RAG) chatbot using LangChain.

-  PromptTemplate: A LangChain utility to define dynamic prompts with
   placeholders.

-  input_variables: These are the dynamic fields (chat_history, context,
   question) that will be filled at runtime.

-  template: The actual prompt structure. It guides the LLM to:

-  Read the chat history (for continuity),

-  Use the retrieved context (from vector store),

-  Answer the latest user question.

✅ Use Case This prompt is ideal for chatbots with memory + retrieval,
where: - chat_history maintains conversation flow, - context comes from
relevant document chunks, - question is the current user query.


.. code:: ipython3

    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

.. code:: ipython3

    def ask_question(query):
        # Retrieve relevant chunks
        docs = db.similarity_search(query, k=5)
        context = format_docs(docs)
        
        # Combine chat history
        history_text = "\n".join(chat_history)
        
        # Format prompt
        final_prompt = prompt.format(chat_history=history_text, context=context, question=query)
        
        # Query LLM
        response = llm.invoke(final_prompt)
        
        # Update chat history
        chat_history.append(f"User: {query}")
        chat_history.append(f"AI: {response}")
        
        return response

.. code:: ipython3

    query1 = "Explain self-attention mechanism"
    answer1 = ask_question(query1)
    print(answer1)
    
    query2 = "How does positional encoding work?"
    answer2 = ask_question(query2)
    print(answer2)
    


.. parsed-literal::

    The self-attention mechanism is a crucial component of the Transformer model, allowing it to attend to different parts of the input sequence and weigh their importance for generating the output. It's a way for the model to understand relationships between different tokens or positions in the input.
    
    In simple terms, self-attention works by computing a weighted sum of the values from different positions in the input, where the weights are learned during training. The input is split into three components: queries (Q), keys (K), and values (V). The model then computes attention scores between each query and key, which represents how relevant each pair is to each other.
    
    The attention mechanism consists of two main steps:
    
    1. **Attention calculation**: The model computes the attention weights (α) by taking the dot product of Q and K, and applying a softmax function to get the normalized weights.
    2. **Weighted sum**: The model takes the weighted sum of the values V, using the attention weights α.
    
    The output is then computed as a weighted sum of the values from different positions in the input, where the weights are learned during training.
    
    In the context of the provided figures (Figure 3, Figure 4, and Figure 5), self-attention helps the model understand long-distance dependencies between tokens. For example, in Figure 3, the attention mechanism is able to attend to a distant dependency of the verb "making" in layer 5 of 6.
    
    Self-attention has several advantages over traditional recurrent neural networks (RNNs) and convolutional neural networks (CNNs):
    
    1. **Parallelization**: Self-attention allows for parallel processing of different parts of the input sequence, making it more efficient.
    2. **Scalability**: Self-attention can handle longer sequences than RNNs or CNNs, which are limited by their sequential nature.
    3. **Interpretability**: The attention weights provide insights into which parts of the input are relevant to the output.
    
    The sinusoidal positional encoding (Equation 1) is used in the self-attention mechanism to encode the position information. This allows the model to easily learn to attend by relative positions, as mentioned in the paper.
    
    I hope this explanation helps you understand the self-attention mechanism!
    In this case, the question is already answered in detail. However, I will provide a concise summary and highlight the key points.
    
    **Positional Encoding**
    
    The positional encoding is used to inject information about the relative or absolute position of tokens in a sequence into the model. This is necessary because self-attention mechanisms do not have a natural notion of order or position.
    
    **How it Works**
    
    The positional encoding uses sine and cosine functions of different frequencies to represent each dimension of the input embeddings. The formula for positional encoding is:
    
    P E(pos,2i) = sin(pos/100002i/dmodel )
    P E(pos,2i+1) = cos(pos/100002i/dmodel )
    
    Where `pos` is the position, and `i` is the dimension.
    
    **Key Points**
    
    * Positional encoding is used to provide information about token positions in a sequence.
    * It uses sine and cosine functions of different frequencies to represent each dimension.
    * The positional encodings have the same dimension as the embeddings (dmodel), so they can be summed together.
    
    Let me know if you'd like any further clarification!
    

.. code:: ipython3

    query3 = "Encoder and Decoder Stacks?"
    answer3 = ask_question(query2)
    print(answer2)
    


.. parsed-literal::

    In this case, the question is already answered in detail. However, I will provide a concise summary and highlight the key points.
    
    **Positional Encoding**
    
    The positional encoding is used to inject information about the relative or absolute position of tokens in a sequence into the model. This is necessary because self-attention mechanisms do not have a natural notion of order or position.
    
    **How it Works**
    
    The positional encoding uses sine and cosine functions of different frequencies to represent each dimension of the input embeddings. The formula for positional encoding is:
    
    P E(pos,2i) = sin(pos/100002i/dmodel )
    P E(pos,2i+1) = cos(pos/100002i/dmodel )
    
    Where `pos` is the position, and `i` is the dimension.
    
    **Key Points**
    
    * Positional encoding is used to provide information about token positions in a sequence.
    * It uses sine and cosine functions of different frequencies to represent each dimension.
    * The positional encodings have the same dimension as the embeddings (dmodel), so they can be summed together.
    
    Let me know if you'd like any further clarification!
    

.. code:: ipython3

    query4 = "Conclusion"
    answer4 = ask_question(query4)
    print(answer4)
    docs = retriever.invoke(question)
    for d in docs:
        print(d.page_content[:400])
    


.. parsed-literal::

    It seems we've reached the end of our conversation! To summarize, we discussed self-attention mechanisms and their application in natural language processing. We covered the following topics:
    
    1. **Self-Attention Mechanism**: Self-attention is a crucial component of the Transformer model that allows it to attend to different parts of the input sequence and weigh their importance for generating the output.
    2. **Positional Encoding**: Positional encoding is used to inject information about the relative or absolute position of tokens in a sequence into the model. This is necessary because self-attention mechanisms do not have a natural notion of order or position.
    3. **Why Self-Attention**: Self-attention was used in this particular work because it addresses three key desiderata for mapping one variable-length sequence to another: parallelization, scalability, and interpretability.
    
    We also reviewed some figures and papers related to self-attention mechanisms and neural machine translation.
    
    If you have any further questions or need additional clarification on any of these topics, feel free to ask!
    4To illustrate why the dot products get large, assume that the components of q and k are independent random
    variables with mean 0 and variance 1. Then their dot product, q · k = Pdk
    i=1 qiki, has mean 0 and variance dk.
    4
    Table 4: The Transformer generalizes well to English constituency parsing (Results are on Section 23
    of WSJ)
    Parser Training WSJ 23 F1
    Vinyals & Kaiser el al. (2014) [37] WSJ only, discriminative 88.3
    Petrov et al. (2006) [29] WSJ only, discriminative 90.4
    Zhu et al. (2013) [40] WSJ only, discriminative 90.4
    Dyer et al. (2016) [8] WSJ only, discriminative 91.7
    Transformer (4 layers) WSJ only, disc
    2017.
    [19] Yoon Kim, Carl Denton, Luong Hoang, and Alexander M. Rush. Structured attention networks.
    In International Conference on Learning Representations, 2017.
    [20] Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In ICLR, 2015.
    [21] Oleksii Kuchaiev and Boris Ginsburg. Factorization tricks for LSTM networks. arXiv preprint
    arXiv:1703.10722, 2017.
    [22] Zhouhan Lin, Min
    [37] Vinyals & Kaiser, Koo, Petrov, Sutskever, and Hinton. Grammar as a foreign language. In
    Advances in Neural Information Processing Systems, 2015.
    [38] Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Wolfgang
    Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, et al. Google’s neural machine
    translation system: Bridging the gap between human and machine translation. 
    

.. code:: ipython3

    query4 = "author of pdf "
    answer4 = ask_question(query4)
    print(answer4)
    docs = retriever.invoke(question)
    for d in docs:
        print(d.page_content[:400])
    


.. parsed-literal::

    The authors of the PDF are not explicitly stated in the provided snippet, but based on the references and citations mentioned, it appears to be a collection of research papers and articles related to natural language processing and neural machine translation.
    
    However, I can provide some information about the specific papers cited:
    
    * The paper "Neural Machine Translation by Jointly Learning to Align and Translate" (Bahdanau et al., 2014) is attributed to Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio.
    * The paper "Massive Exploration of Neural Machine Translation Architectures" (Britz et al., 2017) is attributed to Denny Britz, Anna Goldie, Minh-Thang Luong, and Quoc V. Le.
    
    If you're looking for the authors of the original PDF, I'd be happy to help you investigate further!
    4To illustrate why the dot products get large, assume that the components of q and k are independent random
    variables with mean 0 and variance 1. Then their dot product, q · k = Pdk
    i=1 qiki, has mean 0 and variance dk.
    4
    Table 4: The Transformer generalizes well to English constituency parsing (Results are on Section 23
    of WSJ)
    Parser Training WSJ 23 F1
    Vinyals & Kaiser el al. (2014) [37] WSJ only, discriminative 88.3
    Petrov et al. (2006) [29] WSJ only, discriminative 90.4
    Zhu et al. (2013) [40] WSJ only, discriminative 90.4
    Dyer et al. (2016) [8] WSJ only, discriminative 91.7
    Transformer (4 layers) WSJ only, disc
    2017.
    [19] Yoon Kim, Carl Denton, Luong Hoang, and Alexander M. Rush. Structured attention networks.
    In International Conference on Learning Representations, 2017.
    [20] Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In ICLR, 2015.
    [21] Oleksii Kuchaiev and Boris Ginsburg. Factorization tricks for LSTM networks. arXiv preprint
    arXiv:1703.10722, 2017.
    [22] Zhouhan Lin, Min
    [37] Vinyals & Kaiser, Koo, Petrov, Sutskever, and Hinton. Grammar as a foreign language. In
    Advances in Neural Information Processing Systems, 2015.
    [38] Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Wolfgang
    Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, et al. Google’s neural machine
    translation system: Bridging the gap between human and machine translation. 
    

.. code:: ipython3

    query4 = "Scaled Dot-Product Attention"
    answer4 = ask_question(query4)
    print(answer4)
    docs = retriever.invoke(question)
    for d in docs:
        print(d.page_content[:400])
    


.. parsed-literal::

    The Scaled Dot-Product Attention is a type of attention mechanism used in the Transformer model. It works by computing the dot products of the query with all keys, dividing each by √dk, and applying a softmax function to obtain the weights on the values.
    
    The formula for Scaled Dot-Product Attention is given by:
    
    Attention(Q, K, V) = softmax(QKT
    √dk
    )V
    
    Where Q is the matrix of queries, K is the matrix of keys, V is the matrix of values, and dk is the dimension of the keys. The dot products are scaled by √dk to prevent the dot products from growing too large in magnitude.
    
    The Scaled Dot-Product Attention has several advantages over other attention mechanisms, including additive attention, such as:
    
    * It can be implemented using highly optimized matrix multiplication code.
    * It is much faster and more space-efficient than additive attention.
    * It outperforms additive attention for small values of dk.
    
    However, for larger values of dk, the dot products can grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients. To counteract this effect, scaling the dot products by 1√dk is used.
    4To illustrate why the dot products get large, assume that the components of q and k are independent random
    variables with mean 0 and variance 1. Then their dot product, q · k = Pdk
    i=1 qiki, has mean 0 and variance dk.
    4
    Table 4: The Transformer generalizes well to English constituency parsing (Results are on Section 23
    of WSJ)
    Parser Training WSJ 23 F1
    Vinyals & Kaiser el al. (2014) [37] WSJ only, discriminative 88.3
    Petrov et al. (2006) [29] WSJ only, discriminative 90.4
    Zhu et al. (2013) [40] WSJ only, discriminative 90.4
    Dyer et al. (2016) [8] WSJ only, discriminative 91.7
    Transformer (4 layers) WSJ only, disc
    2017.
    [19] Yoon Kim, Carl Denton, Luong Hoang, and Alexander M. Rush. Structured attention networks.
    In International Conference on Learning Representations, 2017.
    [20] Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In ICLR, 2015.
    [21] Oleksii Kuchaiev and Boris Ginsburg. Factorization tricks for LSTM networks. arXiv preprint
    arXiv:1703.10722, 2017.
    [22] Zhouhan Lin, Min
    [37] Vinyals & Kaiser, Koo, Petrov, Sutskever, and Hinton. Grammar as a foreign language. In
    Advances in Neural Information Processing Systems, 2015.
    [38] Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Wolfgang
    Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, et al. Google’s neural machine
    translation system: Bridging the gap between human and machine translation. 
    

