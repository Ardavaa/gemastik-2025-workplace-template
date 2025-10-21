from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import EditKnowledge
from LangchainSpil import add_user_isue_to_db  # Mengimpor skrip Anda

# Inisialisasi aplikasi Flask dan aktifkan CORS
app = Flask(__name__)
CORS(app)

# --- Rute untuk Pengetahuan (Knowledge) ---

@app.route('/knowledge', methods=['GET'])
def get_all_knowledge():
    """Endpoint untuk mendapatkan semua data pengetahuan."""
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
    """Endpoint untuk menambah pengetahuan baru."""
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
    """Endpoint untuk mengedit pengetahuan yang sudah ada."""
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
    """Endpoint untuk menghapus pengetahuan berdasarkan id_docs."""
    try:
        EditKnowledge.delete_by_id_docs_knowledge(id_docs)
        return jsonify({"message": f"Pengetahuan dengan id_docs {id_docs} berhasil dihapus"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Rute untuk Video Tutorial ---

@app.route('/videos', methods=['GET'])
def get_all_videos():
    """Endpoint untuk mendapatkan semua video tutorial."""
    try:
        df = EditKnowledge.get_all_data_as_dataframe()
        if df.empty:
            return jsonify([])
        return jsonify(df.to_dict(orient='records'))
    except Exception as e:
        return jsonify({"error": f"Terjadi kesalahan internal: {str(e)}"}), 500

@app.route('/videos', methods=['POST'])
def add_video_route():
    """Endpoint untuk menambah video tutorial baru."""
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
    """Endpoint untuk mengedit video tutorial yang sudah ada."""
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
    """Endpoint untuk menghapus video tutorial berdasarkan ID-nya."""
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

@app.route('/')
def index():
    """
    Serves the main index.html file when the user visits the root URL.
    """
    
    return render_template('knowledge_kontrol.html')


if __name__ == '__main__':
    # Menjalankan server di port 5000 agar bisa diakses dari frontend
    # Remember to replace 'your_mysql_user', 'your_mysql_password', 'your_database_name'
    # in the create_db_connection function if you're using the placeholder.
    app.run(host='0.0.0.0', port=5000, debug=True)
