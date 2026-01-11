# Medical Report Reader - Health Assistant

A free, AI-powered web application that helps users understand their medical reports by providing simple explanations, translations, and personalized health recommendations.

## Features

- 📄 **OCR Text Extraction** - Extract text from medical report images
- 🌍 **Multi-Language Support** - Support for 15+ languages including English, Hindi, Spanish, French, and more
- 📖 **Simple Explanations** - Convert complex medical terms into easy-to-understand language
- 🥗 **Diet Recommendations** - Personalized diet advice based on health parameters
- 🏃 **Exercise & Yoga Tips** - Customized exercise routines and yoga practices
- ⚠️ **Prevention Guidance** - Learn what to avoid to maintain better health
- ✅ **Benefits Information** - Understand the benefits of following health recommendations
- 🆓 **100% Free** - No signup required, completely free to use

## Tech Stack

- **Backend:** Python with FastAPI
- **Frontend:** HTML5, CSS3, Vanilla JavaScript
- **OCR:** Tesseract OCR via pytesseract
- **Styling:** Custom CSS with green medical theme

## Installation

### Prerequisites

- Python 3.8 or higher
- Tesseract OCR installed on your system

### Installing Tesseract OCR

#### Windows
1. Download the installer from: https://github.com/UB-Mannheim/tesseract/wiki
2. Install Tesseract OCR
3. Add Tesseract to your system PATH or update the path in `main.py`:
   ```python
   pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
   ```

#### macOS
```bash
brew install tesseract
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

### Setup Instructions

1. **Clone or download this repository**

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create the static directory** (if not already present)
   ```bash
   mkdir static
   ```
   Make sure the following files are in the `static` folder:
   - `index.html`
   - `styles.css`
   - `script.js`

5. **Run the application**
   ```bash
   python main.py
   ```
   
   Or using uvicorn directly:
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

6. **Open your browser**
   Navigate to: `http://localhost:8000`

## Project Structure

```
medical-report-reader/
├── main.py                 # FastAPI backend application
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── static/                # Frontend files
    ├── index.html         # Main HTML page
    ├── styles.css         # Styling with green medical theme
    └── script.js          # Frontend JavaScript logic
```

## API Endpoints

### Health Check
- **GET** `/api/health`
- Returns: Server health status

### OCR Text Extraction
- **POST** `/api/ocr`
- Body: FormData with image file
- Returns: Extracted text from medical report

### Report Analysis
- **POST** `/api/analyze-report`
- Body: `{"report_text": "...", "language": "en"}`
- Returns: Simplified explanation of the report

### Health Recommendations
- **POST** `/api/recommendations`
- Body: `{"report_text": "...", "language": "en"}`
- Returns: Diet, lifestyle, exercise, yoga tips, things to avoid, and benefits

### Translation
- **POST** `/api/translate`
- Body: `{"text": "...", "target_language": "hi"}`
- Returns: Translated text

### Supported Languages
- **GET** `/api/languages`
- Returns: List of supported languages with codes

## AI Integration ✅

**NOW INTEGRATED!** The application uses OpenRouter API with Google's Gemini 2.0 Flash model for:

- **Report Analysis**: AI-powered understanding of medical reports with simple explanations
- **Health Recommendations**: Personalized diet, exercise, yoga, and lifestyle advice
- **Translation**: Multi-language support for all content
- **Smart Prompting**: Anti-hallucination prompts ensure accurate, safe responses

### API Configuration

The application uses OpenRouter API which provides access to multiple AI models. Current setup:
- **Model**: Google Gemini 2.0 Flash (Free tier)
- **API Key**: Stored in `.env` file (never commit this!)
- **Features**: Report analysis, recommendations, translations

### Environment Variables

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your-api-key-here
OPENROUTER_MODEL=google/gemini-2.0-flash-exp:free
```

**Note**: The `.env` file is already configured. Keep your API key secure and never share it publicly.

## Configuration

### Tesseract OCR Path
If Tesseract is not in your system PATH, update `main.py`:

```python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

### CORS Settings
Update the CORS middleware in `main.py` to restrict origins in production:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Update this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Important Notice

⚠️ **Medical Disclaimer:** This application is for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of qualified health providers with any questions regarding medical conditions.

## Features in Development

- 🔄 Comprehensive Health Assistant (Coming Soon)
- 📊 Personalized Health Dashboard
- 📈 Track Health Metrics Over Time
- 🤖 AI Health Coach
- 💊 Medication Reminders
- 📱 Mobile App

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Support

For support, email support@healthreader.com or raise an issue in the repository.

## Acknowledgments

- FastAPI for the excellent web framework
- Tesseract OCR for text extraction
- Font Awesome for icons
- The open-source community

---

**Made with ❤️ for better health awareness**

*Empowering everyone to understand their health better*
