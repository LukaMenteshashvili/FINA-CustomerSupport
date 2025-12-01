// fina_framer_widget.js

(function () {
  // 1) CONFIG
  const CONFIG = {
    // If window.FINA_BACKEND_URL is set by the page, use that.
    // Otherwise default to local dev.
    backendUrl: window.FINA_BACKEND_URL || "http://127.0.0.1:8000/chat",
    title: "FINA Connect",
    subtitle: "Learn about FINA LLC and IRP",
    quickReplies: [
      "Tell me about FINA LLC",
      "Show me your products",
      "Meet the leadership team",
      "Request a demo"
    ]
  };

  function initWidget() {
    // Avoid double init
    if (document.getElementById("fina-chatbox")) return;

    // 2) STYLES (Botpress-like dark theme, scroll fixed)
    const style = document.createElement("style");
    style.innerHTML = `
      /* launcher */
      #fina-launcher {
        position: fixed;
        bottom: 24px;
        right: 24px;
        width: 56px;
        height: 56px;
        border-radius: 999px;
        background: #ff530fff;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        box-shadow: 0 18px 40px rgba(0,0,0,0.45);
        z-index: 999999;
      }
      #fina-launcher-icon {
        width: 30px;
        height: 30px;
        border-radius: 8px;
        background: radial-gradient(circle at 0 0, #ff6427ff, #ff470fff);
        display: flex;
        align-items: center;
        justify-content: center;
        color: #f9fafb;
        font-weight: 700;
        font-size: 18px;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      }

      /* chat window */
      #fina-chatbox {
        position: fixed;
        bottom: 96px;
        right: 24px;
        width: 380px;
        height: 560px;
        border-radius: 18px;
        background: #020617;
        color: #e5e7eb;
        box-shadow: 0 24px 60px rgba(0,0,0,0.70);
        display: none;
        flex-direction: column;
        overflow: hidden;
        z-index: 999999;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      }

      /* make internal layout flex so scroll works */
      #fina-chatbox-inner {
        display: flex;
        flex-direction: column;
        height: 100%;
      }

      /* header */
      #fina-header {
        height: 64px;
        padding: 12px 16px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #020617;
        border-bottom: 1px solid #1f2937;
      }
      #fina-header-left {
        display: flex;
        align-items: center;
        gap: 10px;
      }
      #fina-header-avatar {
        width: 32px;
        height: 32px;
        border-radius: 999px;
        background: radial-gradient(circle at 0 0, #ff6427ff, #ff470fff);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
        font-weight: 700;
        color: #f9fafb;
      }
      #fina-header-text {
        display: flex;
        flex-direction: column;
      }
      #fina-header-text strong {
        font-size: 15px;
        color: #f9fafb;
      }
      #fina-header-text span {
        font-size: 12px;
        color: #9ca3af;
      }
      #fina-header-refresh {
        cursor: pointer;
        font-size: 16px;
        color: #9ca3af;
        padding: 4px;
      }

      /* body */
      #fina-body {
        flex: 1 1 auto;
        display: flex;
        flex-direction: column;
        background: radial-gradient(circle at top left, #111827, #020617);
      }

      #fina-quick-replies {
        padding: 12px 16px 4px 16px;
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
      }
      .fina-quick-btn {
        border-radius: 999px;
        border: none;
        cursor: pointer;
        padding: 6px 12px;
        font-size: 12px;
        background: #f97316;
        color: #111827;
        font-weight: 500;
        white-space: nowrap;
      }

      #fina-messages {
        flex: 1 1 auto;
        padding: 4px 16px 8px 16px;
        overflow-y: auto;
        scrollbar-width: thin;
        scrollbar-color: #4b5563 #020617;
      }
      #fina-messages::-webkit-scrollbar {
        width: 6px;
      }
      #fina-messages::-webkit-scrollbar-track {
        background: #020617;
      }
      #fina-messages::-webkit-scrollbar-thumb {
        background: #4b5563;
        border-radius: 999px;
      }

      .fina-row {
        display: flex;
        margin-top: 8px;
      }
      .fina-row.bot {
        justify-content: flex-start;
        gap: 8px;
      }
      .fina-row.user {
        justify-content: flex-end;
      }
      .fina-bot-avatar-small {
        width: 22px;
        height: 22px;
        border-radius: 999px;
        background: radial-gradient(circle at 0 0, #ff6427ff, #ff470fff);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 12px;
        color: #f9fafb;
        flex-shrink: 0;
        margin-top: 4px;
      }
      .fina-bubble {
        max-width: 80%;
        padding: 10px 12px;
        font-size: 13px;
        line-height: 1.5;
        border-radius: 14px;
        white-space: pre-wrap;
      }
      .fina-bot-msg {
        background: #111827;
        color: #e5e7eb;
        border-bottom-left-radius: 4px;
      }
      .fina-user-msg {
        background: #f97316;
        color: #111827;
        border-bottom-right-radius: 4px;
      }
      .fina-typing {
        font-size: 11px;
        color: #9ca3af;
        margin-top: 4px;
        margin-left: 32px;
      }

      /* footer */
      #fina-footer {
        padding: 8px 12px 6px 12px;
        border-top: 1px solid #1f2937;
        background: #020617;
        display: flex;
        flex-direction: column;
        gap: 6px;
      }
      #fina-input-row {
        display: flex;
        align-items: center;
        gap: 8px;
      }
      #fina-input {
        flex: 1 1 auto;
        border-radius: 999px;
        border: 1px solid #374151;
        background: #020617;
        padding: 8px 12px;
        color: #e5e7eb;
        font-size: 13px;
        outline: none;
      }
      #fina-input::placeholder {
        color: #6b7280;
      }
      #fina-send-btn {
        border-radius: 999px;
        width: 32px;
        height: 32px;
        border: none;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        background: #f97316;
        color: #020617;
        font-size: 16px;
      }
      #fina-footer-powered {
        font-size: 11px;
        color: #6b7280;
        text-align: center;
      }
      #fina-footer-powered span {
        color: #f97316;
      }
    `;
    document.head.appendChild(style);

    // 3) DOM elements
    const launcher = document.createElement("div");
    launcher.id = "fina-launcher";
    launcher.innerHTML = `<div id="fina-launcher-icon">F</div>`;

    const chat = document.createElement("div");
    chat.id = "fina-chatbox";
    chat.innerHTML = `
      <div id="fina-chatbox-inner">
        <div id="fina-header">
          <div id="fina-header-left">
            <div id="fina-header-avatar">F</div>
            <div id="fina-header-text">
              <strong>${CONFIG.title}</strong>
              <span>${CONFIG.subtitle}</span>
            </div>
          </div>
          <div id="fina-header-refresh">&#10227;</div>
        </div>
        <div id="fina-body">
          <div id="fina-quick-replies"></div>
          <div id="fina-messages"></div>
        </div>
        <div id="fina-footer">
          <div id="fina-input-row">
            <input id="fina-input" placeholder="Type your message..." />
            <button id="fina-send-btn">&#10148;</button>
          </div>
          <div id="fina-footer-powered">
            <span>⚡</span> powered by FINA AI
          </div>
        </div>
      </div>
    `;

    document.body.appendChild(launcher);
    document.body.appendChild(chat);

    const messagesEl = chat.querySelector("#fina-messages");
    const inputEl = chat.querySelector("#fina-input");
    const sendBtn = chat.querySelector("#fina-send-btn");
    const refreshBtn = chat.querySelector("#fina-header-refresh");
    const quickEl = chat.querySelector("#fina-quick-replies");

    // 4) Conversation helpers
    function addQuickReplies() {
      quickEl.innerHTML = "";
      CONFIG.quickReplies.forEach(text => {
        const btn = document.createElement("button");
        btn.className = "fina-quick-btn";
        btn.textContent = text;
        btn.onclick = () => handleUserInput(text);
        quickEl.appendChild(btn);
      });
    }

    function addBotMessage(text) {
      const row = document.createElement("div");
      row.className = "fina-row bot";
      const avatar = document.createElement("div");
      avatar.className = "fina-bot-avatar-small";
      avatar.textContent = "F";
      const bubble = document.createElement("div");
      bubble.className = "fina-bubble fina-bot-msg";
      bubble.textContent = text;
      row.appendChild(avatar);
      row.appendChild(bubble);
      messagesEl.appendChild(row);
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    function addUserMessage(text) {
      const row = document.createElement("div");
      row.className = "fina-row user";
      const bubble = document.createElement("div");
      bubble.className = "fina-bubble fina-user-msg";
      bubble.textContent = text;
      row.appendChild(bubble);
      messagesEl.appendChild(row);
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    function setTyping(visible) {
      let t = chat.querySelector(".fina-typing");
      if (visible) {
        if (!t) {
          t = document.createElement("div");
          t.className = "fina-typing";
          t.textContent = "FINA assistant is thinking...";
          messagesEl.appendChild(t);
        }
      } else if (t) {
        t.remove();
      }
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    function resetConversation() {
      messagesEl.innerHTML = "";
      addQuickReplies();
      addBotMessage(
        "Hello and welcome! I'm here to help you learn about FINA LLC, our regulatory technology solutions, and how we support digital transformation for regulatory bodies and enterprises.\n\nHow can I assist you today?"
      );
    }

    async function handleUserInput(text) {
      if (!text) return;
      addUserMessage(text);
      setTyping(true);
      try {
        const res = await fetch(CONFIG.backendUrl, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: text })
        });
        const data = await res.json();
        setTyping(false);
        addBotMessage(data.reply || "I could not get a reply from the server.");
      } catch (err) {
        console.error(err);
        setTyping(false);
        addBotMessage("There was an error contacting the server.");
      }
    }

    async function sendMessage() {
      const text = (inputEl.value || "").trim();
      if (!text) return;
      inputEl.value = "";
      handleUserInput(text);
    }

    // 5) Event bindings
    launcher.onclick = () => {
      const open = chat.style.display === "flex";
      chat.style.display = open ? "none" : "flex";
      if (!open) {
        chat.style.flexDirection = "column";
        inputEl.focus();
        if (!messagesEl.hasChildNodes()) {
          resetConversation();
        }
      }
    };

    sendBtn.onclick = sendMessage;
    inputEl.addEventListener("keypress", e => {
      if (e.key === "Enter") sendMessage();
    });

    refreshBtn.onclick = () => {
      resetConversation();
    };
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initWidget);
  } else {
    initWidget();
  }
})();
