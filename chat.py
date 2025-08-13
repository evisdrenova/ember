from pkg.proto import assistant_pb2, assistant_pb2_grpc
import grpc
import uuid
import os

GRPC_SERVER_HOST = "192.168.1.22"
GRPC_SERVER_PORT = 8080

_VERBOSE = os.getenv("GRPC_DEBUG", "1") == "1"

_GRPC_OPTIONS = [
    ("grpc.keepalive_time_ms", 20000),
    ("grpc.keepalive_timeout_ms", 10000),
    ("grpc.keepalive_permit_without_calls", 1),
    ("grpc.http2.min_time_between_pings_ms", 10000),
    ("grpc.http2.max_pings_without_data", 0),
    ("grpc.max_receive_message_length", 32 * 1024 * 1024),
    ("grpc.max_send_message_length", 32 * 1024 * 1024),
]

def _verbose(): return _VERBOSE

def _log(msg: str):
    if _verbose():
        print(msg)

def _watch_channel(name: str, channel: grpc.Channel):
    def cb(connectivity):
        _log(f"🔌 [{name}] connectivity: {connectivity.name}")
    channel.subscribe(cb, try_to_connect=True)

def _open_channel():
    target = f"{GRPC_SERVER_HOST}:{GRPC_SERVER_PORT}"
    print(f"🔗 Connecting to gRPC server at {target}")
    channel = grpc.insecure_channel(target, options=_GRPC_OPTIONS)
    _watch_channel("assistant", channel)
    grpc.channel_ready_future(channel).result(timeout=5)
    _log("✅ Channel ready")
    return channel

def grpc_chat_response(text: str, timeout: int = 120) -> str:
    
    try:
        channel = _open_channel()
        stub = assistant_pb2_grpc.AssistantServiceStub(channel)
        session_id = str(uuid.uuid4())
        req = assistant_pb2.ChatRequest(session_id=session_id, message=text, audio_data=b"")
        complete = []
        for resp in stub.Chat(req, timeout=timeout):
            if resp.text_response:
                complete.append(resp.text_response)
            if resp.is_final:
                break
        return "".join(complete).strip()
    finally:
        channel.close()