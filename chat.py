from pkg.proto import assistant_pb2
from pkg.proto import assistant_pb2_grpc
import grpc
import uuid

GRPC_SERVER_HOST = "192.168.1.20"               
GRPC_SERVER_PORT = 8080


def grpc_chat_response(text: str) -> str:
    """Send text to Go gRPC server and get response"""
    print(f"🔗 Connecting to gRPC server at {GRPC_SERVER_HOST}:{GRPC_SERVER_PORT}")
    
    try:
        # Create gRPC channel
        channel = grpc.insecure_channel(f'{GRPC_SERVER_HOST}:{GRPC_SERVER_PORT}')
        stub = assistant_pb2_grpc.AssistantServiceStub(channel)
        
        # Generate session ID
        session_id = str(uuid.uuid4())
        
        # Create request
        request = assistant_pb2.ChatRequest(
            session_id=session_id,
            message=text,
            audio_data=b'' 
        )
        
        print(f"📤 Sending request: session={session_id}, message='{text}'")
        
        # Create request stream
        def request_generator():
            yield request
        
        # Call the streaming RPC
        responses = stub.Chat(request_generator())
        
        # Get the first response
        for response in responses:
            print(f"📥 Received response: {response.text_response}")
            channel.close()
            return response.text_response
        
        # If no responses received
        channel.close()
        return "No response received from server"
            
    except Exception as e:
        print(f"❌ gRPC error: {e}")
        return f"Sorry, I couldn't connect to the server: {e}"