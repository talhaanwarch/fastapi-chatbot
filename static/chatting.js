// const ws = new WebSocket("ws://127.0.0.1:8003/");
const ws = new WebSocket("wss:rag.talhaanwar.com");

let responseBuffer = '';
let currentMessage = null;
let autoScroll = true;
let isScrolling = false;

ws.onmessage = function(event) {
    const chunk = event.data;
    
    if (chunk === '[END]') {
        finalizeMessage();
        return;
    }

    if (!currentMessage) {
        createAssistantMessage();
    }

    responseBuffer += chunk;
    currentMessage.innerHTML = marked.parse(responseBuffer);
    forceScrollToBottom();
};

function createAssistantMessage() {
    const messages = document.getElementById('messages');
    currentMessage = document.createElement('div');
    currentMessage.className = 'message assistant-message';
    messages.appendChild(currentMessage);
    responseBuffer = '';
}

function finalizeMessage() {
    currentMessage = null;
    responseBuffer = '';
    document.getElementById('typingIndicator').style.display = 'none';
}

function sendMessage(event) {
    event.preventDefault();
    const input = document.getElementById('messageText');
    const message = input.value.trim();
    if (!message) return;

    const userMessage = document.createElement('div');
    userMessage.className = 'message user-message';
    userMessage.textContent = message;
    document.getElementById('messages').appendChild(userMessage);

    document.getElementById('typingIndicator').style.display = 'block';
    
    ws.send(message);
    input.value = '';
    forceScrollToBottom();
}

function forceScrollToBottom() {
    if (!autoScroll) return;
    
    const messages = document.getElementById('messages');
    const scroll = () => {
        messages.scrollTop = messages.scrollHeight;
        isScrolling = false;
    };
    
    if (!isScrolling) {
        isScrolling = true;
        requestAnimationFrame(scroll);
    }
}

let scrollTimeout;
document.getElementById('messages').addEventListener('scroll', () => {
    const messages = document.getElementById('messages');
    const threshold = 50;
    const atBottom = messages.scrollTop + messages.clientHeight + threshold >= messages.scrollHeight;
    
    autoScroll = atBottom;
    
    clearTimeout(scrollTimeout);
    scrollTimeout = setTimeout(() => {
        if (atBottom) autoScroll = true;
    }, 1000);
});

document.getElementById('messageText').addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage(e);
    }
});

window.addEventListener('load', forceScrollToBottom);
