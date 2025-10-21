from typing import Dict, List, Optional
from qdrant_client import QdrantClient, models
import collections
from sentence_transformers import SentenceTransformer
from qdrant_client.models import PointStruct, VectorParams, Distance
import uuid
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, UpdateStatus, PointStruct, Distance, VectorParams
import pandas as pd
import mysql.connector
from mysql.connector import Error
import os 
from dotenv import load_dotenv
load_dotenv()

db_config = {
    'host': os.getenv('DB_HOST'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_NAME')
}
# --- Qdrant Setup ---
knowledgeNameCollection = "BotSpil"
vidNameCollection = "BotSpilTutorialVideo"
client = QdrantClient(host= os.getenv("QDRANT_HOST", "localhost") , port=6333)
model = SentenceTransformer("all-MiniLM-L6-v2")


def get_all_points_and_group_by_id_docs(collection_name: str):
    """Mengambil semua poin dari koleksi Qdrant dan mengelompokkan berdasarkan id_docs."""
    grouped_data = collections.defaultdict(list)
    next_page_offset = None

    try:
        while True:
            points, next_page_offset = client.scroll(
                collection_name=collection_name,
                with_payload=True,
                with_vectors=False, 
                offset=next_page_offset
            )
            for point in points:
                id_docs = point.payload.get("id_docs")
                if id_docs is not None:
                    grouped_data[id_docs].append(point)
            if not next_page_offset:
                break
        return dict(grouped_data)

    except Exception as e:
        print(f"Terjadi kesalahan: {e}")
        return None
    
def convert_to_tabular(grouped_data: dict) -> list[dict]:
    """mengambil data yang diconvert dan mengubahnya menjadi format yang diatur"""
    tabular_data = []
    for id_docs, points in grouped_data.items():
        if not points:
            continue
        title = points[0].payload.get("konteks", "Tidak ada judul")
        sorted_points = sorted(points, key=lambda p: p.payload.get("urutan", 0))
        all_texts = [p.payload.get("text", "") for p in sorted_points]
        combined_text = "\n".join(filter(None, all_texts))
        tabular_data.append({
            "id_docs": id_docs,
            "title": title,
            "teks": combined_text
        })

    return tabular_data

def chunking_txt(text: str, kontext: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list:
    # Inisialisasi text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    # Bagi teks menjadi potongan/chunks
    chunks = splitter.split_text(text)
    for i in chunks : 
        i = kontext + " " + i

    return chunks

def add_to_qdrant_Chunck_Knowledge(chunks, konteks):
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

def add_to_qdrant_tutorialVid(title, description, links) : 
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

def delete_by_id_docs_knowledge( id_docs_value: str):
    
    print(f"Mencoba menghapus semua data dengan id_docs = '{id_docs_value}'...")
    try:
        # 1. Membuat filter untuk menargetkan payload
        filter_condition = Filter(
            must=[
                FieldCondition(
                    key="id_docs",  # Nama field di payload
                    match=MatchValue(value=id_docs_value) # Nilai yang harus cocok
                )
            ]
        )

        # 2. Menjalankan operasi delete dengan filter tersebut
        result = client.delete(
            collection_name=knowledgeNameCollection,
            points_selector=filter_condition,
            wait=True  # Menunggu hingga operasi selesai di server
        )

        # 3. Memberikan feedback berdasarkan hasil
        if result.status == UpdateStatus.COMPLETED:
            print(f"Berhasil! Operasi penghapusan untuk id_docs '{id_docs_value}' selesai.")
        else:
            print(f"Operasi selesai dengan status: {result.status}")
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")

def edit_knowledge(id_docs : str, new_text: str, kontext: str) :
    delete_by_id_docs_knowledge(id_docs)
    chuncks = chunking_txt(new_text, kontext)
    add_to_qdrant_Chunck_Knowledge(chuncks, kontext)
    print("done edit knowledge")

def add_knowledge(isi, judul) :
    chunks = chunking_txt(isi, kontext=judul)
    add_to_qdrant_Chunck_Knowledge(chunks, kontext=judul)
    print("done add knowledge")

def get_all_data_as_dataframe() -> pd.DataFrame:
    
    print(f"Mengambil semua data dari koleksi '{vidNameCollection}'...")
    
    try:
        # Scroll untuk mendapatkan semua objek PointStruct
        all_points = client.scroll(
            collection_name=vidNameCollection,
            limit=10_000,
            with_payload=True,
            with_vectors=False
        )[0]
        data_list = [{'id': point.id, **point.payload} for point in all_points]

        if not data_list:
            print("⚠️ Koleksi kosong atau tidak ditemukan.")
            return pd.DataFrame()
            
        df = pd.DataFrame(data_list)
        required_cols = ['id', 'title', 'description', 'link']
        for col in required_cols:
            if col not in df.columns:
                df[col] = None
        
        print(f"Berhasil! Ditemukan {len(df)} data.")
        return df[required_cols]

    except Exception as e:
        print(f"Terjadi kesalahan: {e}")
        return pd.DataFrame()

def delete_point_by_id( point_id):
    
    print(f"Mencoba menghapus data dengan ID: '{point_id}'...")
    
    try:
        # Menjalankan operasi delete dengan menargetkan list ID
        result = client.delete(
            collection_name=vidNameCollection,
            points_selector=[point_id], # ID harus berada di dalam sebuah list
            wait=True  # Menunggu hingga operasi selesai
        )
        
        if result.status == UpdateStatus.COMPLETED:
            print(f"Berhasil! Data dengan ID '{point_id}' telah dihapus.")
        else:
            print(f"Operasi selesai dengan status: {result.status}")
            
        return result

    except Exception as e:
        print(f"Terjadi kesalahan: {e}")
        return None

def edit_vid(    point_id: str, 
    new_title: str, 
    new_description: str, 
    new_link: str) :
    delete_point_by_id(point_id)
    add_to_qdrant_tutorialVid(new_title, new_description, new_link)

def create_db_connection():
    """Menciptakan koneksi ke database MySQL."""
    connection = None
    try:
        connection = mysql.connector.connect(**db_config)
        print("Koneksi ke database MySQL berhasil")
    except Error as e:
        print(f"Error '{e}' terjadi saat menyambung ke database")
    return connection


def get_all_errors() -> Optional[List[Dict]]:
    """
    Mengambil semua data isu pengguna dari tabel 'report.eror' dan 
    mengonversinya ke format yang sesuai untuk frontend.
    """
    conn = create_db_connection()
    if conn is None:
        print("Gagal menyambung ke database untuk mengambil error.")
        return None # Or raise an exception, depending on desired error handling

    errors_list = []
    query = "SELECT * FROM report.eror" 
    cursor = conn.cursor(dictionary=True) # Use dictionary=True to get results as dicts

    try:
        cursor.execute(query)
        records = cursor.fetchall()

        for row in records:
            # Map database column names to frontend-expected names
            # and convert boolean isHandle to "Handled"/"Not Handled" string
            errors_list.append({
                "id": str(row['id_eror']), # Ensure ID is a string for frontend consistency
                "errorName": row['nama_eror'],
                "errorDescription": row['deskripsi_eror'],
                "status": "Handled" if row['isHandle'] else "Not Handled"
            })
        return errors_list
    except Error as e:
        print(f"Gagal mengambil data error dari database: {e}")
        return None
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

def add_error(error_name: str, error_description: Optional[str], status: str) -> bool:
    """
    Menambahkan catatan error baru ke dalam tabel 'report.eror' di database.
    Mengonversi status string dari frontend menjadi boolean untuk database.
    """
    conn = create_db_connection()
    if conn is None:
        print("Gagal menyambung ke database untuk menambah error.")
        return False

    # Convert frontend status string ("Handled"/"Not Handled") to boolean for DB
    is_handled_bool = (status.lower() == 'handled')

    query = """
    INSERT INTO report.eror (nama_eror, deskripsi_eror, isHandle) 
    VALUES (%s, %s, %s)
    """
    cursor = conn.cursor()
    try:
        cursor.execute(query, (error_name, error_description, is_handled_bool))
        conn.commit()
        print(f"Sukses: Error '{error_name}' berhasil ditambahkan.")
        return True
    except Error as e:
        print(f"Gagal menambahkan error ke database: {e}")
        return False
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

def edit_error(error_id: str, error_name: str, error_description: Optional[str], status: str) -> bool:
    """
    Mengedit catatan error yang sudah ada di tabel 'report.eror' di database.
    Mengonversi status string dari frontend menjadi boolean untuk database.
    """
    conn = create_db_connection()
    if conn is None:
        print("Gagal menyambung ke database untuk mengedit error.")
        return False

    # Convert frontend status string ("Handled"/"Not Handled") to boolean for DB
    print("edit_error "+status)
    is_handled_bool = 1 if status.lower() == 'handled' else 0

    query = """
    UPDATE report.eror 
    SET nama_eror = %s, deskripsi_eror = %s, isHandle = %s
    WHERE id_eror = %s
    """
    cursor = conn.cursor()
    try:
        cursor.execute(query, (error_name, error_description, is_handled_bool, error_id))
        conn.commit()
        if cursor.rowcount > 0:
            print(f"Sukses: Error dengan ID {error_id} berhasil diperbarui.")
            return True
        else:
            print(f"Peringatan: Error dengan ID {error_id} tidak ditemukan untuk diperbarui.")
            return False
    except Error as e:
        print(f"Gagal mengedit error di database: {e}")
        return False
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

def delete_error_by_id(error_id: str) -> bool:
    """
    Menghapus catatan error dari tabel 'report.eror' berdasarkan ID.
    """
    conn = create_db_connection()
    if conn is None:
        print("Gagal menyambung ke database untuk menghapus error.")
        return False

    query = "DELETE FROM report.eror WHERE id_eror = %s"
    cursor = conn.cursor()
    try:
        cursor.execute(query, (error_id,))
        conn.commit()
        if cursor.rowcount > 0:
            print(f"Sukses: Error dengan ID {error_id} berhasil dihapus.")
            return True
        else:
            print(f"Peringatan: Error dengan ID {error_id} tidak ditemukan untuk dihapus.")
            return False
    except Error as e:
        print(f"Gagal menghapus error dari database: {e}")
        return False
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()


def get_all_issues() -> Optional[List[Dict]]:
    """
    Mengambil semua data isu pengguna dari tabel 'report.user_isue' dan 
    mengonversinya ke format yang sesuai untuk frontend.
    """
    conn = create_db_connection()
    if conn is None:
        print("Gagal menyambung ke database untuk mengambil issue.")
        return None # Or raise an exception, depending on desired issue handling

    issues_list = []
    query = "SELECT * FROM report.user_isue" 
    cursor = conn.cursor(dictionary=True) # Use dictionary=True to get results as dicts

    try:
        cursor.execute(query)
        records = cursor.fetchall()

        for row in records:
            # Map database column names to frontend-expected names
            # and convert boolean isHandle to "Handled"/"Not Handled" string
            issues_list.append({
                "id": str(row['id_isue']), # Ensure ID is a string for frontend consistency
                "issueName": row['nama_isue'],
                "issueDescription": row['deskripsi_isue'],
                "status": "Handled" if row['isHandle'] else "Not Handled"
            })
        return issues_list
    except Error as e:
        print(f"Gagal mengambil data issue dari database: {e}")
        return None
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

def add_issue(issue_name: str, issue_description: Optional[str], status: str) -> bool:
    """
    Menambahkan catatan issue baru ke dalam tabel 'report.user_isue' di database.
    Mengonversi status string dari frontend menjadi boolean untuk database.
    """
    conn = create_db_connection()
    if conn is None:
        print("Gagal menyambung ke database untuk menambah issue.")
        return False

    # Convert frontend status string ("Handled"/"Not Handled") to boolean for DB
    is_handled_bool = (status.lower() == 'handled')

    query = """
    INSERT INTO report.user_isue (nama_isue, deskripsi_isue, isHandle) 
    VALUES (%s, %s, %s)
    """
    cursor = conn.cursor()
    try:
        cursor.execute(query, (issue_name, issue_description, is_handled_bool))
        conn.commit()
        print(f"Sukses: Issue '{issue_name}' berhasil ditambahkan.")
        return True
    except Error as e:
        print(f"Gagal menambahkan issue ke database: {e}")
        return False
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

def edit_issue(issue_id: str, issue_name: str, issue_description: Optional[str], status: str) -> bool:
    """
    Mengedit catatan issue yang sudah ada di tabel 'report.user_isue' di database.
    Mengonversi status string dari frontend menjadi boolean untuk database.
    """
    conn = create_db_connection()
    if conn is None:
        print("Gagal menyambung ke database untuk mengedit issue.")
        return False

    # Convert frontend status string ("Handled"/"Not Handled") to boolean for DB
    print("edit_issue "+status)
    is_handled_bool = 1 if status.lower() == 'handled' else 0

    query = """
    UPDATE report.user_isue 
    SET nama_isue = %s, deskripsi_isue = %s, isHandle = %s
    WHERE id_isue = %s
    """
    cursor = conn.cursor()
    try:
        cursor.execute(query, (issue_name, issue_description, is_handled_bool, issue_id))
        conn.commit()
        if cursor.rowcount > 0:
            print(f"Sukses: issue dengan ID {issue_id} berhasil diperbarui.")
            return True
        else:
            print(f"Peringatan: issue dengan ID {issue_id} tidak ditemukan untuk diperbarui.")
            return False
    except Error as e:
        print(f"Gagal mengedit issue di database: {e}")
        return False
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

def delete_issue_by_id(issue_id: str) -> bool:
    """
    Menghapus catatan issue dari tabel 'report.user_isue' berdasarkan ID.
    """
    conn = create_db_connection()
    if conn is None:
        print("Gagal menyambung ke database untuk menghapus issue.")
        return False

    query = "DELETE FROM report.user_isue WHERE id_isue = %s"
    cursor = conn.cursor()
    try:
        cursor.execute(query, (issue_id,))
        conn.commit()
        if cursor.rowcount > 0:
            print(f"Sukses: issue dengan ID {issue_id} berhasil dihapus.")
            return True
        else:
            print(f"Peringatan: issue dengan ID {issue_id} tidak ditemukan untuk dihapus.")
            return False
    except Error as e:
        print(f"Gagal menghapus issue dari database: {e}")
        return False
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()





