from fastapi import FastAPI
from fastapi.websockets import WebSocket
from fastapi.websockets import ConnectionClosed
import json
import uvicorn

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active_connections.append(ws)
    
    def disconnect(self, ws: WebSocket):
        self.active_connections.remove(ws)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

app = FastAPI()
manager = ConnectionManager()

# Websocket endpoint for client connections (real-time position updates to web GUI)
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            data = await ws.receive_text()
            if data == "connect": 
                print("Client connected")
    except ConnectionClosed:
        manager.disconnect(ws)
    except Exception as e:
        print(f"Error: {e}")
        manager.disconnect(ws)

# Broadcast position updates to all ws clients
@app.post("/position")
async def update_position(position: dict):
    await manager.broadcast(json.dumps(position))

# Start server from main.py to set default port
if __name__ == "__main__": uvicorn.run("web.server.main:app", host="127.0.0.1", port=8001, reload=True)