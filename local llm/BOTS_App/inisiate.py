from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client.models import PointStruct, VectorParams, Distance
import uuid
from dotenv import load_dotenv
import os
from urllib.parse import urlparse

from LangchainSpil import add_user_isue_to_db

load_dotenv()

client = QdrantClient(url="http://localhost:6333")  
model = SentenceTransformer("all-MiniLM-L6-v2")


client.create_collection(
    collection_name="BotSpil",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)
client.create_collection(
    collection_name="BotSpilTutorialVideo",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)
def read_and_chunk_txt(path_to_txt: str, kontext: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list:
    
    with open(path_to_txt, "r", encoding="utf-8") as f:
        raw_text = f.read()
    
    # Inisialisasi text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    # Bagi teks menjadi potongan/chunks
    chunks = splitter.split_text(raw_text)
    for i in chunks : 
        i = kontext + " " + i

    return chunks
gmailChunks = read_and_chunk_txt("./RAG Knowledge/tutorial_gmail.txt", kontext="Gmail Tutorial")
gmeetChunks = read_and_chunk_txt("./RAG Knowledge/tutorial_gmeet.txt", kontext="Gmeet Tutorial")
lmsChunks = read_and_chunk_txt("./RAG Knowledge/tutorial_lms.txt", kontext="lms Tutorial")

def add_to_qdrant_Chunck(chunks, konteks):
    qdrant_points = []
    vectors = model.encode(chunks)
    idDocs = uuid.uuid4()
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        qdrant_points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "id_docs" : idDocs, 
                    "text": chunk,
                    "konteks" : konteks,
                    "urutan" : i                         
                        }
            )
        )
    client.upsert(
        collection_name="BotSpil",
        points=qdrant_points
    )
    print(f"✅ Successfully added {len(chunks)} chunks of {konteks} to Qdrant!")
def add_to_qdrant_tutorialVid( title, description, links) : 
    qdrant_points = []
    vector = model.encode([title + " " + description])[0] 
    qdrant_points.append(
        PointStruct(
        id=str(uuid.uuid4()),
            vector=vector,
            payload={
                    "description": description,
                    "title": title,
                    "link": links
            }
        )
    )
    client.upsert(
        collection_name="BotSpilTutorialVideo",
        points=qdrant_points
    )
    print(f"✅ Successfully added    tutorial videos of {title} to Qdrant!")

print("Adding dummy error data to the database...")

# Dummy Data 1: Handled Error
result1 = add_user_isue_to_db(
    nama_isue="Error 1",
    deskripsi_isue="User tidak dapat menggunakan tombol 'Submit' di halaman profil.",
    isHandle=True # This corresponds to "Handled"
)
print(result1)

# Dummy Data 2: Not Handled (example of an LLM-reported issue)
result2 = add_user_isue_to_db(
    nama_isue="LLM: Query Ambiguity",
    deskripsi_isue="LLM gagal memahami maksud pengguna: 'Apa itu SPILL HELPER dan bagaimana cara kerjanya?'",
    isHandle=False # This corresponds to "Not Handled"
)
print(result2)

# Dummy Data 3: Another Not Handled Error
result3 = add_user_isue_to_db(
    nama_isue="LLM: Knowledge Gap",
    deskripsi_isue="LLM tidak menemukan informasi tentang 'kebijakan pengembalian dana'.",
    isHandle=False
)
print(result3)

print("\nDummy data addition complete. Check your MySQL Workbench or DBeaver to verify.")
print("Then, restart your Flask backend and refresh your frontend to see the data.")
add_to_qdrant_Chunck(gmailChunks, konteks="Gmail Tutorial")
add_to_qdrant_Chunck(gmeetChunks, konteks="Gmeet Tutorial")
add_to_qdrant_Chunck(lmsChunks, konteks="lms Tutorial")
add_to_qdrant_tutorialVid("Tutorial Mengirim Email", "Belajar cara mengirim email pada aplikasi gmail", "https://drive.google.com/file/d/1iHKpnLKT8kcGjK_ZYprO39km6esMuw7R/view?usp=sharing")
add_to_qdrant_tutorialVid("Tutorial membuat meet pada Google meet", "Belajar cara membuat meet pada aplikasi Google meet", "https://drive.google.com/file/d/19AxlEbWWYrf_pALcHbTgkegruiC4AEES/view?usp=drive_link")
