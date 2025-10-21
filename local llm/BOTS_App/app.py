# --- Imports Gabungan ---
# Diambil dari kedua file (app.py dan knowledge_control.py)
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import logging
import base64
import EditKnowledge  # Digunakan oleh kedua file
from LangchainSpil import router_chain, add_user_isue_to_db # Mengimpor kedua fungsi

# --- Inisialisasi Aplikasi Tunggal ---
# Cukup satu kali inisialisasi
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

# --- Rute dari app.py ---

@app.route('/')
def index():
    """
    Menyajikan halaman utama untuk chat (index.html).
    """
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """
    Menangani permintaan chat dari frontend.
    """
    try:
        data = request.json
        message_text = data.get('message', '')
        base64_image = data.get('image', None)

        agent_input_value = {
            "text_query": message_text,
            "image_base64": base64_image
        }
        
        response = router_chain.invoke(agent_input_value)
        bot_response = response.get('output', 'Maaf, terjadi masalah saat memproses respons.')
        logging.info(f"Agent response: {bot_response}")

        return jsonify({'response': bot_response})

    except Exception as e:
        logging.error(f"Terjadi kesalahan di endpoint /chat: {e}", exc_info=True)
        return jsonify({'error': 'Terjadi kesalahan internal pada server.'}), 500

# --- Rute dari knowledge_control.py ---

@app.route('/control') # <-- PENTING: Diubah dari '/' menjadi '/control'
def knowledge_control_index():
    """
    Menyajikan halaman panel kontrol pengetahuan (knowledge_kontrol.html).
    """
    return render_template('knowledge_kontrol.html')

# --- Rute untuk Pengetahuan (Knowledge) ---
@app.route('/knowledge', methods=['GET'])
def get_all_knowledge():
    try:
        grouped_data = EditKnowledge.get_all_points_and_group_by_id_docs(EditKnowledge.knowledgeNameCollection)
        if grouped_data is None:
            return jsonify({"error": "Gagal mengambil data dari Qdrant"}), 500
        tabular_data = EditKnowledge.convert_to_tabular(grouped_data)
        return jsonify(tabular_data)
    except Exception as e:
        return jsonify({"error": f"Terjadi kesalahan internal: {str(e)}"}), 500

@app.route('/knowledge', methods=['POST'])
def add_knowledge_route():
    data = request.json
    if not data or 'title' not in data or 'text' not in data:
        return jsonify({"error": "Judul atau teks tidak boleh kosong"}), 400
    try:
        EditKnowledge.add_knowledge(data['text'], data['title'])
        return jsonify({"message": "Pengetahuan berhasil ditambahkan"}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/knowledge/<id_docs>', methods=['PUT'])
def edit_knowledge_route(id_docs):
    data = request.json
    if not data or 'title' not in data or 'text' not in data:
        return jsonify({"error": "Judul atau teks tidak boleh kosong"}), 400
    try:
        EditKnowledge.edit_knowledge(id_docs, data['text'], data['title'])
        return jsonify({"message": f"Pengetahuan dengan id_docs {id_docs} berhasil diperbarui"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/knowledge/<id_docs>', methods=['DELETE'])
def delete_knowledge_route(id_docs):
    try:
        EditKnowledge.delete_by_id_docs_knowledge(id_docs)
        return jsonify({"message": f"Pengetahuan dengan id_docs {id_docs} berhasil dihapus"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Rute untuk Video Tutorial ---
@app.route('/videos', methods=['GET'])
def get_all_videos():
    try:
        df = EditKnowledge.get_all_data_as_dataframe()
        if df.empty:
            return jsonify([])
        return jsonify(df.to_dict(orient='records'))
    except Exception as e:
        return jsonify({"error": f"Terjadi kesalahan internal: {str(e)}"}), 500

@app.route('/videos', methods=['POST'])
def add_video_route():
    data = request.json
    if not data or 'title' not in data or 'description' not in data or 'link' not in data:
        return jsonify({"error": "Judul, deskripsi, atau link tidak boleh kosong"}), 400
    try:
        EditKnowledge.add_to_qdrant_tutorialVid(data['title'], data['description'], data['link'])
        return jsonify({"message": "Video berhasil ditambahkan"}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/videos/<point_id>', methods=['PUT'])
def edit_video_route(point_id):
    data = request.json
    if not data or 'title' not in data or 'description' not in data or 'link' not in data:
        return jsonify({"error": "Judul, deskripsi, atau link tidak boleh kosong"}), 400
    try:
        EditKnowledge.edit_vid(point_id, data['title'], data['description'], data['link'])
        return jsonify({"message": f"Video dengan id {point_id} berhasil diperbarui"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/videos/<point_id>', methods=['DELETE'])
def delete_video_route(point_id):
    try:
        EditKnowledge.delete_point_by_id(point_id)
        return jsonify({"message": f"Video dengan id {point_id} berhasil dihapus"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Rute Baru untuk Error Tidak Bisa Dihandle ---

@app.route('/error', methods=['GET'])
def get_all_errors():
    """Endpoint untuk mendapatkan semua data error yang tidak bisa dihandle."""
    try:
        # Asumsi ada fungsi di EditKnowledge untuk mendapatkan semua error
        # Fungsi ini harus mengembalikan list of dicts, e.g.,
        # [{"id": "err1", "errorName": "Nama Error", "errorDescription": "Deskripsi", "status": "Handled"}]
        errors_data = EditKnowledge.get_all_errors() 
        print(errors_data)
        if errors_data is None:
            return jsonify([]) # Return empty list if no data
        return jsonify(errors_data)
    except Exception as e:
        print(e)
        return jsonify({"error": f"Terjadi kesalahan internal saat mengambil error: {str(e)}"}), 500

@app.route('/error', methods=['POST'])
def add_error_route():
    """Endpoint untuk menambah error baru (misalnya dari laporan LLM)."""
    data = request.json
    # Memastikan data yang diperlukan ada
    if not data or 'errorName' not in data or 'errorDescription' not in data or 'status' not in data:
        return jsonify({"error": "Nama error, deskripsi, atau status tidak boleh kosong"}), 400
    
    try:
        # Asumsi ada fungsi di EditKnowledge untuk menambah error
        # Fungsi ini akan menyimpan error baru ke database (misalnya Qdrant atau database lain)
        EditKnowledge.add_error(data['errorName'], data['errorDescription'], data['status'])
        return jsonify({"message": "Error berhasil ditambahkan"}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/error/<error_id>', methods=['PUT'])
def edit_error_route(error_id):
    """Endpoint untuk mengedit error yang sudah ada (misalnya mengubah status)."""
    data = request.json
    # Memastikan data yang diperlukan ada
    if not data or 'errorName' not in data or 'errorDescription' not in data or 'status' not in data:
        return jsonify({"error": "Nama error, deskripsi, atau status tidak boleh kosong"}), 400

    try:
        # Asumsi ada fungsi di EditKnowledge untuk mengedit error
        print("edit error route: "+data['status'])
        EditKnowledge.edit_error(error_id, data['errorName'], data['errorDescription'], data['status'])
        return jsonify({"message": f"Error dengan id {error_id} berhasil diperbarui"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/error/<error_id>', methods=['DELETE'])
def delete_error_route(error_id):
    """Endpoint untuk menghapus error berdasarkan ID-nya."""
    try:
        # Asumsi ada fungsi di EditKnowledge untuk menghapus error
        EditKnowledge.delete_error_by_id(error_id)
        return jsonify({"message": f"Error dengan id {error_id} berhasil dihapus"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# --- Rute untuk Issue User ---
# --- Rute Baru untuk Issue User ---

@app.route('/issue', methods=['GET'])
def get_all_issues():
    """Endpoint untuk mendapatkan semua data issue yang tidak bisa dihandle."""
    try:
        # Asumsi ada fungsi di EditKnowledge untuk mendapatkan semua issue
        # Fungsi ini harus mengembalikan list of dicts, e.g.,
        # [{"id": "err1", "issueName": "Nama issue", "issueDescription": "Deskripsi", "status": "Handled"}]
        issues_data = EditKnowledge.get_all_issues() 
        print(issues_data)
        if issues_data is None:
            return jsonify([]) # Return empty list if no data
        return jsonify(issues_data)
    except Exception as e:
        print(e)
        return jsonify({"error": f"Terjadi kesalahan internal saat mengambil issue: {str(e)}"}), 500

@app.route('/issue', methods=['POST'])
def add_issue_route():
    """Endpoint untuk menambah issue baru (misalnya dari laporan LLM)."""
    data = request.json
    # Memastikan data yang diperlukan ada
    if not data or 'issueName' not in data or 'issueDescription' not in data or 'status' not in data:
        return jsonify({"issue": "Nama issue, deskripsi, atau status tidak boleh kosong"}), 400
    
    try:
        # Asumsi ada fungsi di EditKnowledge untuk menambah issue
        # Fungsi ini akan menyimpan issue baru ke database (misalnya Qdrant atau database lain)
        EditKnowledge.add_issue(data['issueName'], data['issueDescription'], data['status'])
        return jsonify({"message": "Issue berhasil ditambahkan"}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/issue/<issue_id>', methods=['PUT'])
def edit_issue_route(issue_id):
    """Endpoint untuk mengedit issue yang sudah ada (misalnya mengubah status)."""
    data = request.json
    # Memastikan data yang diperlukan ada
    if not data or 'issueName' not in data or 'issueDescription' not in data or 'status' not in data:
        return jsonify({"issue": "Nama issue, deskripsi, atau status tidak boleh kosong"}), 400

    try:
        # Asumsi ada fungsi di EditKnowledge untuk mengedit issue
        print("edit issue route: "+data['status'])
        EditKnowledge.edit_issue(issue_id, data['issueName'], data['issueDescription'], data['status'])
        return jsonify({"message": f"Issue dengan id {issue_id} berhasil diperbarui"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/issue/<issue_id>', methods=['DELETE'])
def delete_issue_route(issue_id):
    """Endpoint untuk menghapus issue berdasarkan ID-nya."""
    try:
        # Asumsi ada fungsi di EditKnowledge untuk menghapus issue
        EditKnowledge.delete_issue_by_id(issue_id)
        return jsonify({"message": f"Issue dengan id {issue_id} berhasil dihapus"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Menjalankan Server ---
# Cukup satu kali di akhir file
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)