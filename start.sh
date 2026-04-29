#!/bin/bash

echo ""
echo "================================================"
echo "  Mammogram XAI - Starting up..."
echo "================================================"
echo ""

# ------------------------------------------------
# 1. Find Python 3.10+ (prefer highest available version)
# ------------------------------------------------
PYTHON_CMD=""

for v in python3.13 python3.12 python3.11 python3.10; do
    if command -v "$v" &>/dev/null; then
        PYTHON_CMD="$v"
        break
    fi
done

# Fallback: use python3 if it's already 3.10+
if [ -z "$PYTHON_CMD" ]; then
    if command -v python3 &>/dev/null && \
       python3 -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" 2>/dev/null; then
        PYTHON_CMD="python3"
    fi
fi

if [ -z "$PYTHON_CMD" ]; then
    echo "[ERROR] Python 3.10 or later is required but was not found."
    echo ""
    echo "macOS:  brew install python@3.12"
    echo "Linux:  sudo apt install python3.12 python3.12-venv  (Ubuntu/Debian)"
    echo "        sudo dnf install python3.12                   (Fedora)"
    echo ""
    exit 1
fi

PYTHON_VERSION=$("$PYTHON_CMD" --version 2>&1)
echo "[OK] Found $PYTHON_VERSION ($PYTHON_CMD)"

# ------------------------------------------------
# 2. Check Node.js / npm is installed; auto-install if missing
# ------------------------------------------------
if ! command -v npm &>/dev/null; then
    echo "[WARN] Node.js not found. Attempting automatic installation..."
    echo ""
    OS=$(uname -s)
    INSTALLED=0

    if [ "$OS" = "Darwin" ]; then
        if command -v brew &>/dev/null; then
            echo "[SETUP] Running: brew install node"
            brew install node && INSTALLED=1
        else
            echo "[ERROR] Homebrew is not installed — it is required to auto-install Node.js on macOS."
            echo "        Install Homebrew first, then re-run this script:"
            echo "        /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
            echo "        Or install Node.js directly from https://nodejs.org/ (LTS version)."
            exit 1
        fi
    elif [ "$OS" = "Linux" ]; then
        if command -v apt-get &>/dev/null; then
            echo "[SETUP] Running: sudo apt-get install -y nodejs npm"
            sudo apt-get install -y nodejs npm && INSTALLED=1
        elif command -v dnf &>/dev/null; then
            echo "[SETUP] Running: sudo dnf install -y nodejs"
            sudo dnf install -y nodejs && INSTALLED=1
        elif command -v yum &>/dev/null; then
            echo "[SETUP] Running: sudo yum install -y nodejs npm"
            sudo yum install -y nodejs npm && INSTALLED=1
        else
            echo "[ERROR] Could not detect a supported package manager (apt / dnf / yum)."
            echo "        Install Node.js manually from https://nodejs.org/ (LTS version)."
            exit 1
        fi
    else
        echo "[ERROR] Unrecognised OS ($OS). Install Node.js from https://nodejs.org/ (LTS version)."
        exit 1
    fi

    if [ "$INSTALLED" -eq 0 ] || ! command -v npm &>/dev/null; then
        echo ""
        echo "[ERROR] Installation failed or npm is still not on PATH."
        echo "        Install Node.js manually from https://nodejs.org/ (LTS version)."
        exit 1
    fi

    echo "[OK] Node.js installed successfully."
    echo ""
fi

NPM_VERSION=$(npm --version 2>&1)
echo "[OK] Found npm v$NPM_VERSION"

# ------------------------------------------------
# 3. Create virtual environment if it doesn't exist
#    (recreate automatically if built with wrong Python)
# ------------------------------------------------
if [ ! -d ".venv" ]; then
    echo ""
    echo "[SETUP] Creating Python virtual environment with $PYTHON_VERSION..."
    "$PYTHON_CMD" -m venv .venv
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to create virtual environment."
        echo "        Try: sudo apt install python3.12-venv"
        exit 1
    fi
    echo "[OK] Virtual environment created."
else
    if ! .venv/bin/python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" 2>/dev/null; then
        echo "[WARN] Existing .venv was built with an older Python. Recreating with $PYTHON_VERSION..."
        rm -rf .venv
        "$PYTHON_CMD" -m venv .venv
        if [ $? -ne 0 ]; then
            echo "[ERROR] Failed to recreate virtual environment."
            exit 1
        fi
        echo "[OK] Virtual environment recreated with $PYTHON_VERSION."
    else
        echo "[OK] Virtual environment already exists."
    fi
fi

# ------------------------------------------------
# 4. Activate virtual environment
# ------------------------------------------------
source .venv/bin/activate
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to activate virtual environment."
    exit 1
fi
echo "[OK] Virtual environment activated."

# ------------------------------------------------
# 5. Install / update Python dependencies
# ------------------------------------------------
echo ""
echo "[SETUP] Installing Python dependencies (this may take a moment on first run)..."
pip install -r requirements.txt --quiet
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to install Python dependencies."
    exit 1
fi
echo "[OK] Python dependencies ready."

# ------------------------------------------------
# 6. Install frontend dependencies
# ------------------------------------------------
echo ""
echo "[SETUP] Installing frontend dependencies..."
cd src/frontend
npm install --silent
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to install frontend dependencies."
    cd ../..
    exit 1
fi
cd ../..
echo "[OK] Frontend dependencies ready."

# ------------------------------------------------
# 7. Start the FastAPI backend in the background
# ------------------------------------------------
echo ""
echo "[START] Starting FastAPI backend on http://localhost:8000 ..."
source .venv/bin/activate && uvicorn src.api.server:app --port 8000 &
BACKEND_PID=$!

# ------------------------------------------------
# 8. Give the backend a moment to bind its port
# ------------------------------------------------
sleep 3

# ------------------------------------------------
# 9. Start the Vite frontend in the background
# ------------------------------------------------
echo "[START] Starting frontend on http://localhost:5173 ..."
cd src/frontend && npm run dev &
FRONTEND_PID=$!
cd ../..

# ------------------------------------------------
# 10. Wait for frontend then open browser
# ------------------------------------------------
echo ""
echo "[WAIT] Waiting for servers to be ready..."
sleep 4

echo "[OK] Opening browser..."
if command -v xdg-open &>/dev/null; then
    xdg-open http://localhost:5173       # Linux
elif command -v open &>/dev/null; then
    open http://localhost:5173           # macOS
fi

echo ""
echo "================================================"
echo "  Both servers are running."
echo ""
echo "  Backend:   http://localhost:8000"
echo "  Frontend:  http://localhost:5173"
echo "  API docs:  http://localhost:8000/docs"
echo ""
echo "  Press Ctrl+C to shut down both servers."
echo "================================================"
echo ""

# ------------------------------------------------
# 11. Keep the script alive; shut down both servers
#     cleanly when the user presses Ctrl+C
# ------------------------------------------------
trap "echo ''; echo '[STOP] Shutting down...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" SIGINT SIGTERM

wait
