from queue import Queue
from flask import Flask, request, jsonify
from base_env import MyBaseEnv
from typing import List
import threading
app = Flask(__name__)
queue = Queue()
poll_size = 100
queue_lock = threading.Lock()

@app.route('/setup', methods=['POST'])
async def setup_env():
    try:
        global queue
        queue = Queue() 
        return jsonify({"status": "success", "message": "Server setup complete"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/collect', methods=['POST'])
async def collect(items: List[List[str],float]):
    try:
        with queue_lock:
            conversations = (conversation for conversation, _ in items)
            scores = (score for _, score in items)
            queue.put((conversations, scores))
            return jsonify({"status": "success", "message": "Items collected"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    
@app.route('/dispatch', methods=['POST'])
async def dispatch():
    try:
        with queue_lock:
            if queue.size() > poll_size:
                queue_copy = queue.copy()
                queue.clear()
                return jsonify({"status": "success", "queue": queue_copy})
            else:
                return jsonify({"status": "error", "message": "Queue is Not full"}) , 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
