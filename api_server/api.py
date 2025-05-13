from queue import Queue
from flask import Flask, request, jsonify
from typing import List, Tuple
import threading
app = Flask(__name__)
queue = Queue()
poll_size = 100
queue_lock = threading.Lock()
should_collect = False 
active_collection=False
teardown_flag = False

@app.route('/setup', methods=['POST'])
async def setup_env():
    try:
        global queue, should_collect, active_collection, teardown_flag
        should_collect = False
        active_collection = False
        teardown_flag = False
        queue = Queue() 
        return jsonify({"status": "success", "message": "Server setup complete"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/start_collection', methods=['POST'])
def start_collection():
    try:
        global should_collect
        should_collect = True
        return jsonify({
                "status": "success", 
                "message": "Collection started"
            })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/collect', methods=['POST'])
# List of List of final outputs which contains 
# (scenario:str, character1:str, character2:str, conversation:List[dict], rewards:dict)
async def collect(items: List[List[Tuple[str, str, str, List[dict], dict]]]):
    try:
        with queue_lock:
            for group in items:
                queue.put(group)
            
            # After collecting, set should_collect to False
            should_collect = False
            active_collection = False
            return jsonify({"status": "success", "message": "Items collected"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    
@app.route('/dispatch', methods=['POST'])
async def dispatch():
    try:
        with queue_lock:
            if queue.qsize() > poll_size:
                data= []
                for _ in range(poll_size):
                    data.append(queue.get(0))
                active_collection = False
                return jsonify({"status": "success", "data": data})
            else:
                return jsonify({"status": "error", "message": "Queue is Not full"}) , 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    
    
@app.route('/reload', methods=['POST'])
async def reload_model():
    try:
        global should_collect
        with queue_lock:
            remaining_data = []
            while not queue.empty():
                remaining_data.append(queue.get(0))
            
            should_collect = False 
            active_collection = False
            
            return jsonify({
                "status": "success", 
                "message": "Server is reloading",
                "remaining_data": remaining_data if remaining_data else None
            })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/status', methods=['GET']) 
async def check_status_env():
    try:
        with queue_lock:
            return jsonify({
                "can_sample": should_collect,  
                "queue_size": queue.qsize()
            })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    
@app.route('/status_collection', methods=['GET'])
async def check_status_collection():
    try:
        with queue_lock:
            if should_collect:
                global active_collection
                active_collection = True
            return jsonify({
                "status": "ready", 
                "can_sample": active_collection,     
            })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    

@app.route('/notify_teardown', methods=['POST'])
def notify_teardown():
    global teardown_flag
    teardown_flag = True
    return jsonify({"status": "success", "message": "Teardown notified"})

@app.route('/check_teardown', methods=['GET'])
def check_teardown():
    global teardown_flag
    return jsonify({"status": "success", "message": teardown_flag})

if __name__ == '__main__':
    app.run(debug=True)
