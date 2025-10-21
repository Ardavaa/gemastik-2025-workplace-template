from langchain.agents import initialize_agent, Tool
from langchain_community.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.agents import initialize_agent, Tool
from langchain.schema import HumanMessage
from qdrant_client import QdrantClient
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    Optional,
    TypedDict,
    TypeVar,
    Union,
    cast,
    Annotated,
    Sequence, 
    Optional, 
    Union,
    Dict,
    List,
)
from langchain.llms import Ollama
from langchain_core.messages import HumanMessage,AIMessage, SystemMessage
from langchain.chat_models import ChatOpenAI
import requests
from langchain_core.runnables import RunnableConfig
from langchain.schema import AIMessage, ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from bs4 import BeautifulSoup
import os
from langchain.memory import ConversationBufferMemory
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_core.pydantic_v1 import BaseModel, Field
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

os.environ["OPENAI_API_KEY"] = "DUMMY"

# ---------- Define LLM Class ----------

"""class SimpleEndpointChatModel(ChatOpenAI):
    endpoint: str
    tools: Optional[List[Union[Tool, dict]]] = None
    tool_choice: Optional[str] = None
    
    def __init__(self, endpoint: str, **kwargs):
        # Remove 'openai_api_key' if passed by parent defaults
        kwargs.pop('openai_api_key', None)
        super().__init__(endpoint=endpoint, **kwargs)
        self.endpoint = endpoint

    def _generate(
        self,
        messages, 
        stop=None,
        run_manager=None,
        **kwargs
    ):
        api_messages = []
        # --- MULAI PERUBAHAN DI SINI ---
        for msg in messages:
            # Logika yang lebih lengkap untuk menentukan peran
            if isinstance(msg, SystemMessage):
                role = "system"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            elif isinstance(msg, HumanMessage):
                role = "user"
            else:
                # Fallback jika ada tipe pesan lain
                role = "user"

            api_messages.append({"role": role, "content": msg.content})

        payload = {
            "messages": api_messages,
            "mode": "instruct"
        }
        
        if stop:
            payload["stop"] = stop
        if getattr(self, 'tools', None):
            payload["tools"] = [convert_to_openai_tool(tool) for tool in self.tools]

        try:
            response = requests.post(
                self.endpoint,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            content = f"Error: Could not connect to the model endpoint. {e}"
            
        message = AIMessage(content=content)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self):
        return "kobold"
    
    def bind_tools(
        self,
        tools: Sequence[Union[Tool, dict]],
        tool_choice: Optional[str] = None,
        **kwargs,
    ):
        # Create a new instance with same endpoint and any relevant config
        new_model = self.__class__(endpoint=self.endpoint)  # Add other params as needed
        new_model.tools = list(tools)
        new_model.tool_choice = tool_choice
        return new_model"""
# ---------- Tool Setup ----------
# --- setup RAG Knowledge Tools ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
client = QdrantClient(host=os.getenv("QDRANT_HOST", "localhost") , port=6333)
db = Qdrant(client=client, collection_name="BotSpil", embeddings=embeddings, content_payload_key="text")
def qdrant_rag_tool(query: str, top_k: int = 3) -> str:
    results_with_scores = db.similarity_search_with_score(query, k=top_k)
    threshold = 0.5
    filtered_results = [(doc, score) for doc, score in results_with_scores if score >= threshold]
    return "\n".join(doc.page_content for doc, _ in filtered_results)
"""ragTool = Tool(
    name="qdrant_rag",
    func=qdrant_rag_tool,
    description="Berdasarkan kueri pengguna, ambil konteks yang relevan dari penyimpanan vektor Qdrant untuk menjadi dasar jawaban."
)
"""
# --- Setup Vid tutor redtrival --- 

dbvid = Qdrant(client=client, collection_name="BotSpilTutorialVideo", embeddings=embeddings, content_payload_key="link")
def qdrant_rag_tool_vid(query: str, top_k: int = 1) -> str:
    results_with_scores = dbvid.similarity_search_with_score(query, k=top_k)
    threshold = 0.35
    filtered_results = [(doc, score) for doc, score in results_with_scores if score >= threshold]
    if not filtered_results:
        return "No relevant video found."
    first_doc = filtered_results[0][0] 
    video_link = first_doc.metadata.get("link") or first_doc.page_content
    
    return f"Here is the link: {video_link}"
ragVidtutorTool= Tool(
    name="qdrant_rag_VidTutor",
    func=qdrant_rag_tool_vid,
    description="Berdasarkan kueri pengguna, ambil konteks relevan yang berisi judul, deskripsi, dan video tutorial dari penyimpanan vektor Qdrant untuk menjadi dasar jawaban."
)

# --- Setup Web Scraper Structure Tool ---

def fetch_navigation_structure(url):
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, "html.parser")
    
    # Ambil navigasi (menu) utama
    navs = []
    for nav in soup.find_all(["nav"]):
        navs.append(nav.get_text(separator=" | ", strip=True))

    # Ambil button dan link berlabel penting
    buttons = []
    for tag in soup.find_all(["button", "a", "input"]):
        # Cek id/class yang mengandung kata "button" atau "nav" atau "menu"
        attr_str = (tag.get("id") or "") + " " + " ".join(tag.get("class") or [])
        if (
            "button" in attr_str.lower() or
            "nav" in attr_str.lower() or
            "menu" in attr_str.lower()
        ):
            label = tag.get_text(strip=True) or tag.get("aria-label") or tag.get("title") or attr_str
            buttons.append({
                "type": tag.name,
                "label": label,
                "attributes": dict(tag.attrs)
            })
        # Untuk button sederhana (tanpa id/class)
        elif tag.name == "button":
            label = tag.get_text(strip=True) or tag.get("aria-label") or tag.get("title", "")
            buttons.append({
                "type": "button",
                "label": label,
                "attributes": dict(tag.attrs)
            })

    # Kembalikan hasil terstruktur
    return {
        "title": soup.title.string.strip() if soup.title and soup.title.string else "",
        "main_navigation": navs,
        "buttons_and_links": buttons
    }

scraperStructureTool = Tool(
    name="scraper_Structure",
    func=fetch_navigation_structure,
    description="diberikan sebuah URL, ambil struktur web untuk membantumu memahami isi halaman tersebut. Gunakan tool ini jika pengguna bertanya tentang tutorial atau cara menggunakan situs web, atau jika pengguna bertanya tentang struktur situs web. Outputnya akan berupa struktur dari situs web, dengan judul, navigasi utama, dan tombol/tautan beserta atributnya."
)
# --- Setup Web Scraper Content Tool ---
def retrieve_website_content(url: str) -> dict:
    try:
        # Mengambil halaman
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return {"error": f"Gagal mengambil halaman: {e}"}

    # Parsing HTML
    soup = BeautifulSoup(response.text, "html.parser")

    # Judul
    title = soup.title.string.strip() if soup.title else "No title"

    # Header (h1 - h3)
    headings = []
    for tag in soup.find_all(["h1", "h2", "h3"]):
        headings.append({
            "tag": tag.name,
            "text": tag.get_text(strip=True)
        })

    # Paragraphs
    paragraphs = []
    for p in soup.find_all("p"):
        text = p.get_text(strip=True)
        if text:
            paragraphs.append(text) 

    return {
        "url": url,
        "title": title,
        "headings": headings,
        "paragraphs": paragraphs
    }

scraperContentTool = Tool(
    name="scraper_Structure",
    func=retrieve_website_content,
    description="diberikan sebuah URL, ambil konten web untuk membantumu memahami isi halaman tersebut. Gunakan tool ini jika membutuhkan informasi tentang isi situs web. Outputnya akan berupa konten dari situs web tersebut."
)

# --- Setup db Tool ---
def create_db_connection():
    """Menciptakan koneksi ke database MySQL."""
    connection = None
    try:
        connection = mysql.connector.connect(**db_config)
        print("Koneksi ke database MySQL berhasil")
    except Error as e:
        print(f"Error '{e}' terjadi saat menyambung ke database")
    return connection
def parse_error_string(error_text: str) -> dict:
    parsed_data = {}
    lines = error_text.strip().split('\n')
    
    for line in lines:
        if ':' in line:
            # Memisahkan key dan value berdasarkan ':' pertama
            key, value = line.split(':', 1)
            
            # Membersihkan spasi ekstra dari key dan value
            clean_key = key.strip()
            clean_value = value.strip()
            
            # Mengonversi nilai 'isHandle' menjadi boolean
            if clean_key == 'isHandle':
                parsed_data[clean_key] = clean_value.lower() == 'true'
            else:
                parsed_data[clean_key] = clean_value
                
    return parsed_data
# --- setup db eror tool ---
def add_eror_to_db(ans: str) -> str:
    """Menambahkan catatan error baru ke dalam tabel 'eror' di database."""
    parsedData = parse_error_string(ans)
    conn = create_db_connection()
    if conn is None:
        return "Gagal menyambung ke database."
        
    query = """
    INSERT INTO eror (nama_eror, deskripsi_eror, isHandle) 
    VALUES (%s, %s, %s)
    """
    cursor = conn.cursor()
    try:
        cursor.execute(query, (parsedData["nama_eror"], parsedData["deskripsi_eror"], parsedData["isHandle"]))
        conn.commit()
        return f"Sukses: Error '{parsedData['nama_eror']}' berhasil ditambahkan ke database dengan ID: {cursor.lastrowid}"
    except Error as e:
        return f"Gagal menambahkan error ke database: {e}"
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

add_eror_tool = Tool(
    name="add_eror_to_database",
    func=add_eror_to_db,
    description="""diberikan nama dan deskripsi error, tambahkan error tersebut ke database. Gunakan tool ini jika Anda mengalami error yang tidak bisa Anda tangani dan perlu melaporkannya ke developer. berikan input dengan template ini : 
    nama_eror: <error_name>
    deskripsi_eror: <error_description>
    isHandle: False"""
)


# --- setup db isue tool ---
def add_user_isue_to_db(ans: str) -> str:
    """Menambahkan catatan isu pengguna baru ke dalam tabel 'User_isue' di database."""
    parsedData = parse_error_string(ans)
    conn = create_db_connection()
    if conn is None:
        return "Gagal menyambung ke database."

    query = """
    INSERT INTO User_isue (nama_isue, deskripsi_isue, isHandle) 
    VALUES (%s, %s, %s)
    """
    cursor = conn.cursor()
    try:
        cursor.execute(query, (parsedData["nama_isue"], parsedData["deskripsi_isue"], parsedData["isHandle"]))
        conn.commit()
        return f"Sukses: Isu pengguna '{parsedData['nama_isue']}' berhasil ditambahkan ke database dengan ID: {cursor.lastrowid}"
    except Error as e:
        return f"Gagal menambahkan isu pengguna ke database: {e}"
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

add_user_isue_tool = Tool(
    name="add_user_issue_to_database",
    func=add_user_isue_to_db,
    description="""diberikan nama dan deskripsi isu, tambahkan isu tersebut ke database. Gunakan tool ini jika Anda mengalami error yang tidak bisa Anda bantu terkait permintaan pengguna dan perlu melaporkannya ke developer. berikan input dengan template ini :
    nama_isue: <issue_name>
    deskripsi_isue: <issue_description>
    isHandle: False"""
)

# ---------- Agent ----------

# --- Setup Agent ---
# just in case we need to fix chat bot respond

systemPromt = """
kamu adalah asisten AI yang akan membantu pengguna dalam menjawab pertanyaan mereka terkait data sains dan machine learning
"""

system_message_prompt = SystemMessagePromptTemplate.from_template(systemPromt)
agent_kwargs = {"system_message": system_message_prompt}
tools = []

# --- initialize agent --- 
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
#llm = SimpleEndpointChatModel(endpoint="http://pe.spil.co.id/kobold/v1/chat/completions")
llm = Ollama(model="@ardava fill this with your model name") 
agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=True,
    memory=memory,
    agent_kwargs=agent_kwargs,
    handle_parsing_errors=True
)
# ---------- Chain ----------
# --- Setup Image Route ---
def analyze_base64_image(image_base64: str) -> str:
    
    try:
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Berikan deskripsi singkat untuk gambar ini, lalu di bawahnya, ekstrak semua teks yang bisa kamu baca dari gambar tersebut."
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                },
            ]
        )
        response_object = llm.invoke([message])
        print(response_object.content)
        return response_object.content

    except Exception as e:
        # Penanganan error umum untuk API atau masalah lainnya
        return f"Terjadi error saat menganalisis gambar: {str(e)}"

class RouterInput(BaseModel):
    text_query: str
    image_base64: Optional[str] = None

def image_processing_flow(inputs: Dict) -> Dict:
    print("--- [ROUTER] Terdeteksi gambar, menjalankan alur analisis gambar... ---")
    analysis_result = analyze_base64_image(inputs['image_base64'])
    enriched_prompt = (
        f"{inputs['text_query']}\n\n"
        f"[Konteks Tambahan dari Gambar]:\n{analysis_result}"
    )
    
    print(f"--- [ROUTER] Mengirim prompt yang diperkaya ke agent utama... ---")
    agent_response = agent.invoke({"input": enriched_prompt})
    
    return agent_response

def check_for_image(inputs: dict) -> bool:
    has_image = "image_base64" in inputs and inputs["image_base64"] is not None
    return has_image
# --- Setup text only Route ---
def text_only_flow(inputs: Dict) -> Dict:
    print("--- [ROUTER] Tidak ada gambar, menjalankan alur teks saja... ---")
    agent_response = agent.invoke({"input": inputs["text_query"]})
    return agent_response
# --- initialize Chain ---
router_chain = RunnableBranch(
    (check_for_image, RunnableLambda(image_processing_flow)), 
    RunnableLambda(text_only_flow)                           
).with_types(input_type=RouterInput)
