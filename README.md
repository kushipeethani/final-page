## Setup

### 1. Clone the repository
git clone https://github.com/Subham999100/AI-Recruitment-Portal.git
cd AI-Recruitment-Portal

### 2. Create a virtual environment
python -m venv venv

### 3. Activate the virtual environment
venv\Scripts\activate

### 4. Install dependencies
pip install -r requirements.txt

### 5. Create `.env`

Create a `.env` file in the project root:

GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile

Never commit `.env` to GitHub.

## Run the Backend

From the project root:

```bash
uvicorn Backend.main:app --reload
```

The backend will start at: `http://127.0.0.1:8000`
Interactive Swagger Docs: `http://127.0.0.1:8000/docs`

---

## Run the Frontend

From the project root:

```bash
cd frontend
npm install
npm run dev
```

The frontend will start at: `http://localhost:5173`
