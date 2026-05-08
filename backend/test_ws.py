import asyncio
import websockets

HOST = "0.0.0.0"
PORT = 4000


async def handle_client(websocket):
    client_ip = websocket.remote_address[0]
    print(f"Client connected: {client_ip}")

    try:
        async for message in websocket:
            print(f"Received: {message}")

            # Echo message back to client
            response = f"Server received: {message}"
            await websocket.send(response)

    except websockets.exceptions.ConnectionClosed:
        print(f"Client disconnected: {client_ip}")


async def main():
    async with websockets.serve(handle_client, HOST, PORT):
        print(f"WebSocket server running on ws://{HOST}:{PORT}")
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    asyncio.run(main())
