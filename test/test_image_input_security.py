from __future__ import annotations

import asyncio
import base64
import os
import socket
import threading
import unittest
from unittest.mock import patch

from curl_cffi import CurlOpt
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from api import app as app_module
from api import image_inputs


PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEklEQVR4nGPkEpFjYGBgYgADAALmAEAUQs4PAAAAAElFTkSuQmCC"
)


class FakeResponse:
    def __init__(self, status_code: int, headers: dict[str, str], chunks: list[bytes] | None = None) -> None:
        self.status_code = status_code
        self.headers = headers
        self._chunks = chunks if chunks is not None else [PNG_BYTES]
        self.closed = False

    def iter_content(self):
        yield from self._chunks

    def close(self) -> None:
        self.closed = True


class FakeSession:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []
        self.closed = False

    def get(self, _url: str, **kwargs: object) -> FakeResponse:
        self.calls.append(kwargs)
        return self.response

    def close(self) -> None:
        self.closed = True


class ImageInputSecurityTests(unittest.TestCase):
    def test_data_url_requires_valid_raster_and_matching_mime(self) -> None:
        encoded = base64.b64encode(PNG_BYTES).decode("ascii")
        image = image_inputs._decode_data_url(f"data:image/png;base64,{encoded}")
        self.assertEqual(image[2], "image/png")
        with self.assertRaisesRegex(HTTPException, "MIME type does not match"):
            image_inputs._decode_data_url(f"data:image/jpeg;base64,{encoded}")
        with self.assertRaisesRegex(HTTPException, "valid raster"):
            image_inputs._decode_base64_image(base64.b64encode(b"not an image").decode("ascii"), "x.png", "image/png")

    def test_private_literal_and_private_dns_destinations_are_rejected(self) -> None:
        for url in ("http://127.0.0.1/image.png", "http://[::1]/image.png", "http://169.254.169.254/image.png"):
            with self.subTest(url=url), self.assertRaisesRegex(HTTPException, "public IP"):
                image_inputs._download_image_url(url)
        with patch.object(image_inputs.socket, "getaddrinfo", return_value=[(2, 1, 6, "", ("192.168.1.5", 80))]):
            with self.assertRaisesRegex(HTTPException, "public IP"):
                image_inputs._download_image_url("http://private.example/image.png")

    def test_direct_remote_fetch_pins_public_dns_and_checks_each_redirect(self) -> None:
        responses = [
            FakeResponse(302, {"location": "https://public.example/final.png"}),
            FakeResponse(200, {"content-type": "image/png"}),
        ]
        sessions = [FakeSession(response) for response in responses]
        with (
            patch.object(image_inputs.socket, "getaddrinfo", return_value=[(2, 1, 6, "", ("93.184.216.34", 80))]),
            patch.object(image_inputs.requests, "Session", side_effect=sessions) as session_factory,
        ):
            result = image_inputs._download_image_url("http://public.example/start")
        self.assertEqual(result[2], "image/png")
        self.assertEqual(session_factory.call_count, 2)
        for session_call, session in zip(session_factory.call_args_list, sessions):
            self.assertFalse(session.calls[0]["allow_redirects"])
            self.assertTrue(session.calls[0]["stream"])
            self.assertIn(CurlOpt.RESOLVE, session_call.kwargs["curl_options"])
            self.assertTrue(session.closed)
        self.assertTrue(responses[0].closed)
        self.assertTrue(responses[1].closed)

    def test_redirect_to_private_dns_is_rejected_before_second_fetch(self) -> None:
        response = FakeResponse(302, {"location": "https://private.example/final.png"})
        with (
            patch.object(
                image_inputs.socket,
                "getaddrinfo",
                side_effect=[
                    [(2, 1, 6, "", ("93.184.216.34", 80))],
                    [(2, 1, 6, "", ("10.0.0.4", 80))],
                ],
            ),
            patch.object(image_inputs.requests, "Session", return_value=FakeSession(response)) as session_factory,
            self.assertRaisesRegex(HTTPException, "public IP"),
        ):
            image_inputs._download_image_url("http://public.example/start")
        self.assertEqual(session_factory.return_value.calls[0]["allow_redirects"], False)
        self.assertEqual(session_factory.return_value.calls[0]["stream"], True)
        self.assertTrue(response.closed)

    def test_configured_http_proxy_receives_pinned_public_connect_target(self) -> None:
        ready = threading.Event()
        captured: list[bytes] = []
        proxy_port: list[int] = []

        def proxy_server() -> None:
            listener = socket.socket()
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            port = listener.getsockname()[1]
            proxy_port.append(port)
            ready.set()
            connection, _ = listener.accept()
            with connection:
                connection.settimeout(3)
                data = b""
                while b"\r\n\r\n" not in data:
                    chunk = connection.recv(4096)
                    if not chunk:
                        break
                    data += chunk
                captured.append(data)
                connection.sendall(b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n\r\n")
            listener.close()

        thread = threading.Thread(target=proxy_server, daemon=True)
        thread.start()
        ready.wait(3)
        proxy = f"http://127.0.0.1:{proxy_port[0]}"
        with (
            patch.object(image_inputs.proxy_settings, "build_session_kwargs", return_value={"proxy": proxy}),
            patch.object(image_inputs.socket, "getaddrinfo", return_value=[(2, 1, 6, "", ("93.184.216.34", 443))]),
            self.assertRaises(HTTPException),
        ):
            image_inputs._download_image_url("https://public.example/image.png")
        thread.join(3)
        self.assertIn(b"CONNECT 93.184.216.34:443", captured[0])

    def test_streamed_download_closes_and_rejects_body_over_50mb(self) -> None:
        response = FakeResponse(
            200,
            {"content-type": "image/png"},
            [b"x" * (image_inputs.MAX_IMAGE_REFERENCE_BYTES + 1)],
        )
        with self.assertRaisesRegex(HTTPException, "50MB"):
            image_inputs._read_response_bytes(response)
        self.assertTrue(response.closed)

    def test_input_count_and_aggregate_bytes_are_bounded(self) -> None:
        source = (PNG_BYTES, "image.png", "image/png")
        with self.assertRaisesRegex(HTTPException, "at most 16"):
            image_inputs._sources_from_value([source[0]] * (image_inputs.MAX_IMAGE_INPUT_COUNT + 1))
        with patch.object(image_inputs, "MAX_IMAGE_INPUT_BYTES", len(PNG_BYTES) + 1):
            with self.assertRaisesRegex(HTTPException, "combined image inputs"):
                asyncio.run(image_inputs.read_image_sources([source, source]))

    def test_create_app_registers_boundary_owned_router_and_closed_default_cors(self) -> None:
        with patch.dict(os.environ, {"CHATGPT2API_CORS_ORIGINS": ""}, clear=False):
            app = app_module.create_app()
        paths = {getattr(route, "path", "") for route in app.routes}
        for route in app.routes:
            original_router = getattr(route, "original_router", None)
            if original_router is not None:
                paths.update(getattr(item, "path", "") for item in original_router.routes)
        self.assertIn("/api/workbench/ai/accounts", paths)
        self.assertIn("/api/image-tasks/{task_id}/images/{index}", paths)
        cors = next(item for item in app.user_middleware if item.cls is CORSMiddleware)
        self.assertEqual(cors.kwargs["allow_origins"], [])
        self.assertTrue(any(
            getattr(item.kwargs.get("dispatch"), "__module__", "") == "api.external_images"
            for item in app.user_middleware
        ))
        with TestClient(app) as client:
            response = client.get(
                "/api/workbench/ai/accounts",
                headers={"Authorization": "Bearer invalid", "X-Workbench-Image-Client": "1"},
            )
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
