# FINA Framer Embed

This folder contains a drop-in HTML snippet you can paste into a Framer "Custom Code" block to ship the FINA chatbot UI.

## Files
- `fina_framer_widget.html` – standalone HTML/CSS/JS that matches the design in the mock.

## Usage
1. Deploy the backend with `uvicorn api_server:app --host 0.0.0.0 --port 8000` (or your preferred host). The FastAPI service exposes `POST /chat` and `POST /reindex`.
2. In Framer, add a Custom Code block and paste the contents of `fina_framer_widget.html`.
3. Set a `data-api-url` attribute on the root `<div class="chat-shell" id="fina-chat" data-api-url="https://your-host/chat">` pointing at your deployed backend.
4. Publish and the widget will call your API directly.
