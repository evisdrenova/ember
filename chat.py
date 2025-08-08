from pkg.proto import assistant_pb2
from pkg.proto import assistant_pb2_grpc
import grpc
import uuid
import time
import threading
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

def _log(msg: str):
    if _verbose():
        print(msg)

def _verbose():
    return _VERBOSE

def _watch_channel(name: str, channel: grpc.Channel):
    def cb(connectivity):
        _log(f"🔌 [{name}] connectivity: {connectivity.name}")
    channel.subscribe(cb, try_to_connect=True)

def _open_channel():
    target = f"{GRPC_SERVER_HOST}:{GRPC_SERVER_PORT}"
    print(f"🔗 Connecting to gRPC server at {target}")
    channel = grpc.insecure_channel(target, options=_GRPC_OPTIONS)
    _watch_channel("assistant", channel)
    try:
        grpc.channel_ready_future(channel).result(timeout=5)
        _log("✅ Channel ready")
    except Exception as e:
        _log(f"❌ Channel not ready: {e}")
        channel.close()
        raise
    return channel

def grpc_chat_response(text: str):
    """
    Stream text chunks from the server.
    Yields only real text, but prints heavy debug to stdout.
    """
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

        # IMPORTANT: keep an iterator alive (not a list that immediately gets GC'd)
        def req_iter():
            # You can expand this later to send more client messages
            yield request
            _log("➡️  Client finished sending requests (half-close)")

        responses = stub.Chat(req_iter(), timeout=120)

        got_any_msg = False
        got_any_text = False

        for i, response in enumerate(responses, start=1):
            got_any_msg = True
            # Raw debug
            try:
                # ListFields shows which fields are actually present on the wire
                fields = response.ListFields()
                _log(f"📦 Response #{i} type={type(response)} fields={[(f[0].name, f[1]) for f in fields]}")
            except Exception as e:
                _log(f"⚠️ Could not dump ListFields(): {e}")

            # Try known text fields
            chunk = ""
            try:
                chunk = getattr(response, "text_response", "")
            except Exception as e:
                _log(f"⚠️ error accessing text_response: {e}")

            if not chunk:
                # fallback: if your proto ever changes
                for alt in ("delta", "content", "text", "output_text"):
                    v = getattr(response, alt, "")
                    if v:
                        _log(f"ℹ️  Found text in alternate field '{alt}'")
                        chunk = v
                        break

            _log(f"📥 Received chunk (len={len(chunk)}): {chunk[:120]!r}")
            if chunk:
                got_any_text = True
                yield chunk

            if getattr(response, "is_final", False):
                _log("🏁 Server set is_final=True, ending read loop.")
                break

        if not got_any_msg:
            print("⚠️ No response messages received from server at all.")
            # Don’t yield fallback text to TTS; just log
        elif not got_any_text:
            print("⚠️ Received messages but none had text fields set (check proto field tags & stubs).")

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
