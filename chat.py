from pkg.proto import assistant_pb2
from pkg.proto import assistant_pb2_grpc
import grpc
import uuid

GRPC_SERVER_HOST = "192.168.1.22"
GRPC_SERVER_PORT = 8080

# gRPC channel options: keepalive + bigger message sizes just in case
_GRPC_OPTIONS = [
    ("grpc.keepalive_time_ms", 20000),                 # send keepalive pings every 20s
    ("grpc.keepalive_timeout_ms", 10000),              # wait 10s for ack
    ("grpc.keepalive_permit_without_calls", 1),        # allow pings when no calls
    ("grpc.http2.min_time_between_pings_ms", 10000),
    ("grpc.http2.max_pings_without_data", 0),
    ("grpc.max_receive_message_length", 32 * 1024 * 1024),
    ("grpc.max_send_message_length", 32 * 1024 * 1024),
]

def _open_channel():
    """Create a ready channel or raise fast."""
    target = f"{GRPC_SERVER_HOST}:{GRPC_SERVER_PORT}"
    print(f"🔗 Connecting to gRPC server at {target}")
    channel = grpc.insecure_channel(target, options=_GRPC_OPTIONS)
    try:
        # Fail fast if server isn't reachable
        grpc.channel_ready_future(channel).result(timeout=3)
    except Exception as e:
        channel.close()
        raise
    return channel

def grpc_chat_response(text: str):
    """
    Stream text chunks from the Go server.
    Yields strings as they arrive so TTS can speak immediately.
    """
    try:
        channel = _open_channel()
        stub = assistant_pb2_grpc.AssistantServiceStub(channel)

        session_id = str(uuid.uuid4())
        request = assistant_pb2.ChatRequest(
            session_id=session_id,
            message=text,
            audio_data=b"",
        )
        print(f"📤 Sending request: session={session_id}, message='{text}'")

        # Single request in a client-stream is fine; server streams responses back.
        responses = stub.Chat(iter([request]), timeout=120)

        got_any = False
        for response in responses:
            # Prefer text_response; fall back to other common fields if your proto differs
            chunk = getattr(response, "text_response", "") \
                 or getattr(response, "delta", "") \
                 or getattr(response, "content", "")

            if chunk:
                got_any = True
                print(f"📥 Received response chunk: '{chunk[:80]}'")
                yield chunk

            # Only stop early if the server explicitly marks final
            if getattr(response, "is_final", False):
                print("🏁 Server marked response as final.")
                break

        channel.close()

        if not got_any:
            print("⚠️ No response chunks received from server.")
            yield "No response received from server"

    except grpc.RpcError as e:
        # Surface a readable error to the TTS so you hear it if needed
        msg = f"gRPC error: {e.code().name}: {e.details()}"
        print(f"❌ {msg}")
        yield msg
    except Exception as e:
        msg = f"Sorry, something went wrong: {e}"
        print(f"❌ {msg}")
        yield msg

def grpc_chat_response_collect(text: str) -> str:
    """Convenience: collect the whole streamed response into one string."""
    return "".join(grpc_chat_response(text))
