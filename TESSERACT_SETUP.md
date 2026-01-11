# Tesseract OCR Installation Guide

## Windows Installation

1. **Download Tesseract OCR:**
   - Go to: https://github.com/UB-Mannheim/tesseract/wiki
   - Download the latest Windows installer (e.g., `tesseract-ocr-w64-setup-5.3.3.20231005.exe`)

2. **Install Tesseract:**
   - Run the installer
   - Default installation path: `C:\Program Files\Tesseract-OCR`
   - Make sure to select "Add to PATH" during installation

3. **Verify Installation:**
   ```bash
   tesseract --version
   ```

4. **If Not in PATH:**
   - Open `main.py`
   - Find line ~105 and uncomment:
   ```python
   pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
   ```

## Alternative: Use AI-Based OCR (No Installation Needed)

The application now automatically falls back to AI-based text extraction if Tesseract is not available.

**Note:** AI-based OCR may have limitations with image analysis depending on the model used.

## Troubleshooting

### "Failed to extract text from image"
- **Cause:** Tesseract not installed or image quality issues
- **Solution:** 
  1. Install Tesseract OCR (see above)
  2. Ensure image is clear, well-lit, and in focus
  3. Try scanning/photographing the report again with better lighting

### "OCR service unavailable"
- **Cause:** Both Tesseract and AI fallback failed
- **Solution:**
  1. Install Tesseract OCR
  2. Check your internet connection (for AI fallback)
  3. Verify API key is correctly set in `.env`

### Permission Errors
- Run command prompt as Administrator
- Ensure Tesseract installation directory has proper permissions

## Quick Test

After installation, test with this command:
```bash
tesseract --version
```

You should see output like:
```
tesseract 5.3.3
```

## Need Help?

If you continue to have issues:
1. Make sure your medical report image is clear and readable
2. Try uploading in JPG or PNG format
3. Check that the text in the image is not too small or blurry
