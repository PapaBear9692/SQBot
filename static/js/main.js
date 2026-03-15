// main.js

// ===== DOM ELEMENTS =====
const chatMessages = document.getElementById("chatMessages");
const chatForm = document.getElementById("chatForm");
const messageInput = document.getElementById("messageInput");
const sendButton = document.getElementById("sendButton");
const newChatButton = document.getElementById("newChatButton");
const historyList = document.getElementById("historyList");
const containerEl = document.querySelector(".container");
const sidebarToggle = document.querySelector(".sidebar-toggle");

const BOT_AVATAR = containerEl?.dataset.botAvatar || "";
const USER_AVATAR = containerEl?.dataset.userAvatar || "";
const LOCAL_STORAGE_KEY = "medicine_chat_conversations_v1";
const BOT_DISCLAIMER_TEXT =
  "Please Remember, I am not a doctor, I can only provide product related information. Please consult a doctor for diagnosis, treatment, or dosing decisions.";

// =====================
// Streaming speed controls (global)
// Tweak from DevTools, e.g. STREAM_WPS = 14
// =====================
window.STREAM_ENABLED = true;        // master switch
window.STREAM_MODE = "word";         // "word" | "char"
window.STREAM_WPS = 200;              // words per second (word mode)
window.STREAM_CPS = 500;              // chars per second (char mode)
window.STREAM_PUNCT_PAUSE_MS = 120;  // extra pause after . ! ? …
window.STREAM_NEWLINE_PAUSE_MS = 50; // extra pause after newline
window.STREAM_MIN_DELAY_MS = 10;     // floor
window.STREAM_MD_RENDER_INTERVAL_MS = 60; // re-render markdown every N ms during streaming

function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
}

function isPunctEnding(token) {
    return /[.!?…]$/.test(token);
}

/**
 * Streams tokens INTO element as they arrive (SSE), animating character-by-character.
 * Accepts a state object that gets updated with tokens as they arrive.
 * Animation continuously renders what's in the buffer, character by character.
 * Creates message element and removes typing indicator on first token.
 */
async function streamTokensAsArriving(streamState, typingIndicator = null, receivedIntent = null) {
    if (!streamState) return "";

    let cancelled = false;
    const cancelRef = () => { cancelled = true; };

    const useMarkdown = (typeof marked !== "undefined");
    const mode = window.STREAM_MODE || "word";
    const renderEvery = Math.max(10, Number(window.STREAM_MD_RENDER_INTERVAL_MS) || 60);
    
    let displayedUpTo = 0;
    let lastRenderAt = 0;
    let messageCreated = false;
    let el = null;

    const createMessageAndRemoveTyping = () => {
        if (messageCreated) return;
        messageCreated = true;
        
        // Remove typing indicator
        if (typingIndicator && typingIndicator.isConnected) {
            typingIndicator.remove();
        }
        
        // Create message element
        const { content } = createBotMessageElementEmpty(receivedIntent);
        el = content;
    };

    const renderNow = (upToIndex) => {
        if (cancelled || !el) return;

        const shouldStick =
            chatMessages
                ? (chatMessages.scrollTop + chatMessages.clientHeight >= chatMessages.scrollHeight - 40)
                : true;

        const displayedText = streamState.buffer.substring(0, upToIndex);
        if (useMarkdown) {
            el.innerHTML = marked.parse(displayedText);
        } else {
            el.textContent = displayedText;
        }

        if (shouldStick) scrollToBottom();
    };

    const maybeRender = (upToIndex) => {
        const now = Date.now();
        if (now - lastRenderAt >= renderEvery) {
            lastRenderAt = now;
            renderNow(upToIndex);
        }
    };

    if (mode === "char") {
        const cps = Number(window.STREAM_CPS) || 500;
        const baseDelay = Math.max(window.STREAM_MIN_DELAY_MS, Math.round(1000 / cps));

        while (!cancelled) {
            // Check if there's new content to animate
            if (displayedUpTo >= streamState.buffer.length && streamState.done) {
                break;
            }

            if (displayedUpTo < streamState.buffer.length) {
                // Create message on first character
                if (!messageCreated) createMessageAndRemoveTyping();
                
                // Fast mode: dump rest without animation delays
                if (streamState.fastMode) {
                    displayedUpTo = streamState.buffer.length;
                    renderNow(displayedUpTo);
                    // Wait for more content or completion
                    if (!streamState.done) {
                        await sleep(50);
                    }
                } else {
                    // Normal mode: character by character with delays
                    const char = streamState.buffer[displayedUpTo];
                    displayedUpTo++;

                    maybeRender(displayedUpTo);

                    let extra = 0;
                    if (/[.!?…]/.test(char)) extra += Number(window.STREAM_PUNCT_PAUSE_MS) || 0;
                    if (char === "\n") extra += Number(window.STREAM_NEWLINE_PAUSE_MS) || 0;

                    await sleep(baseDelay + extra);
                }
            } else {
                await sleep(5);
            }
        }
    } else {
        // Word mode
        const wps = Number(window.STREAM_WPS) || 200;
        const baseDelay = Math.max(window.STREAM_MIN_DELAY_MS, Math.round(1000 / wps));

        while (!cancelled) {
            // Check if there's new content to animate
            if (displayedUpTo >= streamState.buffer.length && streamState.done) {
                break;
            }

            if (displayedUpTo < streamState.buffer.length) {
                // Create message on first character
                if (!messageCreated) createMessageAndRemoveTyping();
                
                // Fast mode: dump rest without animation delays
                if (streamState.fastMode) {
                    displayedUpTo = streamState.buffer.length;
                    renderNow(displayedUpTo);
                    // Wait for more content or completion
                    if (!streamState.done) {
                        await sleep(50);
                    }
                } else {
                    // Normal mode: word by word with delays
                    const remaining = streamState.buffer.substring(displayedUpTo);
                    const wordMatch = remaining.match(/(\s+|\S+)/);

                    if (wordMatch) {
                        const token = wordMatch[0];
                        displayedUpTo += token.length;

                        maybeRender(displayedUpTo);

                        let extra = 0;
                        const trimmed = token.trim();
                        if (trimmed && isPunctEnding(trimmed)) extra += Number(window.STREAM_PUNCT_PAUSE_MS) || 0;
                        if (token.includes("\n")) extra += Number(window.STREAM_NEWLINE_PAUSE_MS) || 0;

                        await sleep(baseDelay + extra);
                    } else {
                        await sleep(5);
                    }
                }
            } else {
                await sleep(5);
            }
        }
    }

    // Final render to ensure complete formatting
    if (el) {
        renderNow(streamState.buffer.length);
    }

    return streamState.buffer;
}



// streamIntoElement removed - using streamTokensAsArriving for real-time streaming instead


// ===== STATE =====
let conversations = [];
let currentConversationId = null;

// ✅ FIX: track in-flight requests per conversation (prevents input staying disabled)
const pendingByConversation = Object.create(null);
function updateInputDisabledState() {
    const pending = pendingByConversation[currentConversationId] || 0;
    const shouldDisable = pending > 0;
    if (sendButton) sendButton.disabled = shouldDisable;
    if (messageInput) messageInput.disabled = shouldDisable;
}

// ===== UTILITIES =====
function saveConversations() {
    try {
        localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(conversations));
    } catch (e) {
        console.error("Failed to save conversations:", e);
    }
}

function loadConversations() {
    try {
        const raw = localStorage.getItem(LOCAL_STORAGE_KEY);
        if (!raw) return [];

        const parsed = JSON.parse(raw);

        // Ensure timestamps exist on old data
        return parsed.map((c) => {
            const now = Date.now();
            return {
                ...c,
                createdAt: c.createdAt || now,
                updatedAt: c.updatedAt || now
            };
        });
    } catch (e) {
        console.error("Failed to parse stored conversations:", e);
        return [];
    }
}

function createConversation(title = "New chat") {
    const id = Date.now().toString();
    const now = Date.now();

    const convo = {
        id,
        title,
        messages: [], // { role: 'user' | 'bot', content: string }
        createdAt: now,
        updatedAt: now
    };

    conversations.unshift(convo); // newest at top

    // Optional: limit number of saved conversations
    if (conversations.length > 50) {
        conversations = conversations.slice(0, 50);
    }

    currentConversationId = id;
    saveConversations();
    renderSidebar();
    renderConversation();

    // ✅ FIX: update input state for the newly selected chat
    updateInputDisabledState();
}

function getCurrentConversation() {
    return conversations.find((c) => c.id === currentConversationId) || null;
}

// Helper (NEW): safely get a conversation by ID
function getConversationById(conversationId) {
    return conversations.find((c) => c.id === conversationId) || null;
}

// Helper (NEW): add message to a specific conversation (prevents “wrong chat” bug)
function addMessageToConversationById(conversationId, role, text, intent = null) {
  const convo = getConversationById(conversationId);
  if (!convo) return;

  convo.messages.push({ role, content: text, intent: intent || null });
  convo.updatedAt = Date.now();
  saveConversations();
}


// Short title based on first user question
function updateConversationTitleIfNeeded(userText) {
    const convo = getCurrentConversation();
    if (!convo) return;

    if (!convo.title || convo.title === "New chat") {
        const trimmed = userText.trim();
        convo.title =
            trimmed.length > 40 ? trimmed.slice(0, 37) + "..." : trimmed || "New chat";
        convo.updatedAt = Date.now();
        saveConversations();
        renderSidebar();
    }
}

function scrollToBottom() {
    if (!chatMessages) return;
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Format timestamps for sidebar
function formatTimestamp(ts) {
    if (!ts) return "";
    const date = new Date(ts);
    if (Number.isNaN(date.getTime())) return "";

    const now = new Date();
    const isSameDay = date.toDateString() === now.toDateString();

    if (isSameDay) {
        return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    }
    return date.toLocaleDateString([], { month: "short", day: "numeric" });
}

// ===== RENDERING =====
function clearMessages() {
    if (!chatMessages) return;
    chatMessages.innerHTML = "";
}

// Welcome message (not stored in history, shown on top of every conversation)
function renderWelcomeMessage() {
    if (!chatMessages) return;

    const wrapper = document.createElement("div");
    wrapper.className = "message bot-message";

    const avatar = document.createElement("div");
    avatar.className = "message-avatar bot-avatar";

    if (BOT_AVATAR) {
        const img = document.createElement("img");
        img.src = BOT_AVATAR;
        img.alt = "Bot";
        img.className = "bot-logo-avatar";
        avatar.appendChild(img);
    } else {
        avatar.textContent = "AI";
    }

    const content = document.createElement("div");
    content.className = "message-content";

    const note = document.createElement("span");
    note.className = "message-note";
    note.textContent =
        "⚠️ This service provides product related information only and is not a substitute for professional medical advice. Please consult a registered healthcare professional for medical decisions.";

    const p = document.createElement("p");
    p.textContent =
        "Hello, I’m your medicine information assistant. I can help with general information about medicines, dosage forms and usage guidelines. How may I assist you today?";

    content.appendChild(note);
    content.appendChild(p);

    wrapper.appendChild(avatar);
    wrapper.appendChild(content);

    chatMessages.appendChild(wrapper);
}

function createMessageElement(role, text, intent = null) {
    const wrapper = document.createElement("div");
    wrapper.className = `message ${role === "user" ? "user-message" : "bot-message"}`;

    const avatar = document.createElement("div");
    avatar.className = `message-avatar ${role === "user" ? "user-avatar" : "bot-avatar"}`;

    if (role === "bot") {
        if (BOT_AVATAR) {
            const img = document.createElement("img");
            img.src = BOT_AVATAR;
            img.alt = "Bot";
            img.className = "bot-logo-avatar";
            avatar.appendChild(img);
        } else {
            avatar.textContent = "AI";
        }
    } else {
        if (USER_AVATAR) {
            const img = document.createElement("img");
            img.src = USER_AVATAR;
            img.alt = "User";
            img.className = "user-avatar-img";
            avatar.appendChild(img);
        } else {
            avatar.textContent = "You";
        }
    }

    const content = document.createElement("div");
    content.className = "message-content";

    if (role === "bot" && intent === "SYMPTOM_HELP") {
        const note = document.createElement("span");
        note.className = "message-note";
        note.textContent = BOT_DISCLAIMER_TEXT;
        content.appendChild(note);
    }


    const body = document.createElement("div");
    if (typeof marked !== "undefined") body.innerHTML = marked.parse(text);
    else {
        const p = document.createElement("p");
        p.textContent = text;
        body.appendChild(p);
    }
    content.appendChild(body);


    wrapper.appendChild(avatar);
    wrapper.appendChild(content);
    return wrapper;
}

// Create an EMPTY bot message bubble and return its content element (for streaming)
function createBotMessageElementEmpty(intent = null) {
  const wrapper = document.createElement("div");
  wrapper.className = "message bot-message";

  const avatar = document.createElement("div");
  avatar.className = "message-avatar bot-avatar";
  if (BOT_AVATAR) {
    const img = document.createElement("img");
    img.src = BOT_AVATAR;
    img.alt = "Bot";
    img.className = "bot-logo-avatar";
    avatar.appendChild(img);
  } else {
    avatar.textContent = "AI";
  }

  const content = document.createElement("div");
  content.className = "message-content";

  // ✅ Disclaimer only for SYMPTOM_HELP
  if (intent === "SYMPTOM_HELP") {
    const note = document.createElement("span");
    note.className = "message-note";
    note.textContent = BOT_DISCLAIMER_TEXT;
    content.appendChild(note);
  }

  const streamBody = document.createElement("div");
  content.appendChild(streamBody);

  wrapper.appendChild(avatar);
  wrapper.appendChild(content);

  if (chatMessages) {
    chatMessages.appendChild(wrapper);
    scrollToBottom();
  }

  return { wrapper, content: streamBody };
}

function addMessageToUI(role, text, intent = null) {
  if (!chatMessages) return;
  const el = createMessageElement(role, text, intent);
  chatMessages.appendChild(el);
  scrollToBottom();
}

// Render the messages for the current conversation
function renderConversation() {
    clearMessages();
    renderWelcomeMessage();

    const convo = getCurrentConversation();
    if (!convo) return;

    convo.messages.forEach((msg) => {
        addMessageToUI(msg.role, msg.content, msg.intent || null);
    });
}

// Render sidebar history list
function renderSidebar() {
    if (!historyList) return;
    historyList.innerHTML = "";

    if (!conversations.length) {
        const empty = document.createElement("div");
        empty.className = "history-empty";
        empty.textContent = "No chats yet.";
        historyList.appendChild(empty);
        return;
    }

    conversations.forEach((convo) => {
        const item = document.createElement("button");
        item.className = "history-item";
        if (convo.id === currentConversationId) {
            item.classList.add("active");
        }
        item.dataset.id = convo.id;
        item.setAttribute("type", "button");

        const main = document.createElement("div");
        main.className = "history-main";

        const titleEl = document.createElement("span");
        titleEl.className = "history-title";
        titleEl.textContent = convo.title || "New chat";

        const timeEl = document.createElement("span");
        timeEl.className = "history-time";
        timeEl.textContent = formatTimestamp(convo.updatedAt || convo.createdAt);

        main.appendChild(titleEl);
        main.appendChild(timeEl);

        const deleteEl = document.createElement("span");
        deleteEl.className = "history-delete";
        deleteEl.setAttribute("role", "button");
        deleteEl.setAttribute("aria-label", "Delete chat");
        deleteEl.innerHTML = `
            <svg viewBox="0 0 20 20" aria-hidden="true">
                <path d="M7 4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1h3a1 1 0 1 1 0 2h-1v9a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6H4a1 1 0 1 1 0-2h3zm1 2v9h4V6H8z"></path>
            </svg>
        `;

        item.appendChild(main);
        item.appendChild(deleteEl);
        historyList.appendChild(item);
    });
}

// ===== EVENT HANDLERS =====
async function handleSubmit(event) {
    event.preventDefault();
    if (!messageInput || !sendButton) return;

    const text = messageInput.value.trim();
    if (!text) return;

    // If no current conversation, create one first
    if (!currentConversationId) {
        createConversation();
    }

    // IMPORTANT: pin the conversation ID for this request
    // so responses can't land in a different chat if user switches
    const sendConversationId = currentConversationId;

    // ✅ FIX: mark this conversation as pending + update input state for current chat
    pendingByConversation[sendConversationId] = (pendingByConversation[sendConversationId] || 0) + 1;
    updateInputDisabledState();

    // Add user message (UI + correct conversation)
    addMessageToUI("user", text);
    addMessageToConversationById(sendConversationId, "user", text);
    updateConversationTitleIfNeeded(text);

    messageInput.value = "";
    messageInput.focus();

    // Typing indicator
    const typingDiv = document.createElement("div");
    typingDiv.className = "message bot-message typing-indicator";

    const typingAvatar = document.createElement("div");
    typingAvatar.className = "message-avatar bot-avatar";
    if (BOT_AVATAR) {
        const img = document.createElement("img");
        img.src = BOT_AVATAR;
        img.alt = "Bot";
        img.className = "bot-logo-avatar";
        typingAvatar.appendChild(img);
    } else {
        typingAvatar.textContent = "AI";
    }

    const typingContent = document.createElement("div");
    typingContent.className = "message-content";

    const dots = document.createElement("div");
    dots.className = "typing-dots";
    dots.innerHTML = "<span></span><span></span><span></span>";

    typingContent.appendChild(dots);
    typingDiv.appendChild(typingAvatar);
    typingDiv.appendChild(typingContent);

    chatMessages.appendChild(typingDiv);
    scrollToBottom();

    try {
        const formData = new FormData();
        formData.append("msg", text);
        formData.append("conversation_id", sendConversationId || "");

        const response = await fetch("stream", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        // Handle Server-Sent Events (SSE)
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        
        let targetConversationId = sendConversationId;
        let receivedIntent = null;
        let botContentEl = null;
        
        // State object shared between SSE handler and animation
        const streamState = { buffer: "", done: false, tokenCount: 0, fastMode: false };
        let animationPromise = null;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value, { stream: true });
            const lines = chunk.split("\n");

            for (const line of lines) {
                if (line.startsWith("data: ")) {
                    try {
                        const jsonStr = line.slice(6); // Remove "data: " prefix
                        const event = JSON.parse(jsonStr);

                        if (event.type === "start") {
                            targetConversationId = event.conversation_id || sendConversationId;
                            receivedIntent = event.intent || null;

                            // Only render in UI if user is still viewing that conversation
                            if (currentConversationId === targetConversationId) {
                                // Don't create message element yet - wait for first token
                                // Start animation immediately - message and typing removal happen on first token
                                animationPromise = streamTokensAsArriving(streamState, typingDiv, receivedIntent);
                            }
                        } else if (event.type === "token") {
                            const token = event.content || "";
                            // Add token to buffer - animation will pick it up and render it
                            streamState.buffer += token;
                            
                            // Track token count and switch to fast mode after 50 chunks
                            streamState.tokenCount++;
                            if (streamState.tokenCount > 20) {
                                streamState.fastMode = true;
                            }
                            
                        } else if (event.type === "end") {
                            // Streaming complete
                            targetConversationId = event.conversation_id || sendConversationId;
                            streamState.done = true;
                            
                        } else if (event.type === "error") {
                            const errorContent = event.content || "An error occurred.";
                            streamState.buffer = errorContent;
                            streamState.done = true;
                            throw new Error(errorContent);
                        }
                    } catch (parseErr) {
                        // Skip malformed SSE events
                        if (parseErr instanceof SyntaxError) {
                            continue;
                        }
                        throw parseErr;
                    }
                }
            }
        }

        // Wait for animation to complete
        if (animationPromise) {
            try {
                await animationPromise;
            } catch (e) {
                console.error("Animation error:", e);
                throw e;
            }
        }

        // Store the complete response in the correct conversation
        addMessageToConversationById(targetConversationId, "bot", streamState.buffer);

        // If user switched chats during streaming, update sidebar
        if (currentConversationId !== targetConversationId) {
            renderSidebar();
        }
    } catch (error) {
        console.error("Error:", error);

        const errorMsg = "Sorry, there was an error contacting the server.";

        // Store error in the correct conversation
        addMessageToConversationById(sendConversationId, "bot", errorMsg);

        // Only render in UI if user is still viewing that conversation
        if (currentConversationId === sendConversationId) {
            if (typingDiv && typingDiv.isConnected) typingDiv.remove();

            // Display error message
            const { content } = createBotMessageElementEmpty();
            if (typeof marked !== "undefined") {
                content.innerHTML = marked.parse(errorMsg);
            } else {
                content.textContent = errorMsg;
            }
        } else {
            if (typingDiv && typingDiv.isConnected) typingDiv.remove();
            renderSidebar();
        }
    } finally {
        // ✅ FIX: clear pending flag for this conversation
        pendingByConversation[sendConversationId] = (pendingByConversation[sendConversationId] || 1) - 1;
        if (pendingByConversation[sendConversationId] <= 0) {
            delete pendingByConversation[sendConversationId];
        }

        // ✅ FIX: set disabled state based on currently selected chat
        updateInputDisabledState();

        // Keep your existing focus behavior (only focus if still on same chat)
        if (currentConversationId === sendConversationId) {
            messageInput.focus();
        }
    }
}

function handleNewChat() {
    createConversation();
    if (messageInput) {
        messageInput.value = "";
        messageInput.focus();
    }

    // ✅ FIX: update input state for the newly selected chat
    updateInputDisabledState();
}

function handleHistoryClick(event) {
    const deleteBtn = event.target.closest(".history-delete");
    if (deleteBtn) {
        const parentItem = deleteBtn.closest(".history-item");
        if (!parentItem) return;

        const id = parentItem.dataset.id;
        if (!id) return;

        const confirmed = window.confirm("Delete this chat?");
        if (!confirmed) return;

        conversations = conversations.filter((c) => c.id !== id);

        // Send reset request to server to delete the chat history from memory
        const resetData = new FormData();
        resetData.append("conversation_id", id);
        fetch("reset", {
            method: "POST",
            body: resetData
        }).catch(() => {
            // ignore network errors here, UI is already updated
        });

        if (!conversations.length) {
            currentConversationId = null;
            createConversation();
            return;
        }

        if (!conversations.find((c) => c.id === currentConversationId)) {
            currentConversationId = conversations[0].id;
        }

        saveConversations();
        renderSidebar();
        renderConversation();

        // ✅ FIX: update disabled state after selection changes
        updateInputDisabledState();
        return;
    }

    const btn = event.target.closest(".history-item");
    if (!btn) return;

    const id = btn.dataset.id;
    if (!id || id === currentConversationId) return;

    currentConversationId = id;
    renderSidebar();
    renderConversation();

    // ✅ FIX: update input disabled state for the selected chat
    updateInputDisabledState();
}

function handleSidebarToggle() {
    if (!containerEl) return;
    containerEl.classList.toggle("sidebar-open");
}

// ===== INIT =====
function init() {
    if (!chatForm || !chatMessages) return;

    conversations = loadConversations();

    if (conversations.length) {
        // Use most recent conversation
        currentConversationId = conversations[0].id;
    } else {
        // Create first default conversation
        createConversation();
    }

    renderSidebar();
    renderConversation();

    chatForm.addEventListener("submit", handleSubmit);
    if (newChatButton) {
        newChatButton.addEventListener("click", handleNewChat);
    }
    if (historyList) {
        historyList.addEventListener("click", handleHistoryClick);
    }
    if (sidebarToggle) {
        sidebarToggle.addEventListener("click", handleSidebarToggle);
    }

    // ✅ FIX: initialize disabled state on page load
    updateInputDisabledState();
}

document.addEventListener("DOMContentLoaded", init);
