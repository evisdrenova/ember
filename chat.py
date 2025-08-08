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

def grpc_chat_response(text: str):
    """Unary request -> server-streaming responses. Yields text chunks."""
    channel = None
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

        got_any_msg = False
        got_any_text = False

        responses = stub.Chat(request, timeout=120)
        for i, response in enumerate(responses, start=1):
            got_any_msg = True
            try:
                fields = response.ListFields()
                _log(f"📦 Response #{i} fields={[(f[0].name, f[1]) for f in fields]}")
            except Exception as e:
                _log(f"⚠️ Could not dump ListFields(): {e}")

            chunk = getattr(response, "text_response", "")
            _log(f"📥 Received chunk (len={len(chunk)}): {chunk[:120]!r}")
            if chunk:
                got_any_text = True
                yield chunk

            if getattr(response, "is_final", False):
                _log("🏁 is_final=True; ending")
                break

        if not got_any_msg:
            print("⚠️ No response messages received from server.")
        elif not got_any_text:
            print("⚠️ Got responses but no text_response set (check proto field tags & stubs).")

    except grpc.RpcError as e:
        print(f"❌ gRPC error: {e.code().name}: {e.details()}")
        yield f"g r p c error: {e.code().name}: {e.details()}"
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        yield f"sorry, something went wrong: {e}"
    finally:
        if channel:
            channel.close()

def grpc_chat_response_collect(text: str) -> str:
    return "".join(grpc_chat_response(text))
