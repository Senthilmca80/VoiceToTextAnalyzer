from fastapi import FastAPI, File, UploadFile
import whisper
import spacy
import tempfile
import shutil
import re

from transformers import pipeline

app = FastAPI()

# Load models
whisper_model = whisper.load_model("base")  # Choose "small", "medium", "large" for different trade-offs
nlp = spacy.load("en_core_web_sm")
sentiment_pipeline = pipeline("sentiment-analysis")  # Uses a DistilBERT-based sentiment classifier by default

# List of common filler words
FILLER_WORDS = {"um", "uh", "like", "you know", "so"}

def count_filler_words(text: str) -> int:
    """
    Count occurrences of common filler words in the provided text.
    """
    # Normalize the text to lowercase and use regex to split on whitespace/punctuation
    words = re.findall(r'\w+', text.lower())
    return sum(word in FILLER_WORDS for word in words)

def generate_feedback(text: str, sentiment: dict) -> dict:
    """
    Generate actionable public speaking feedback based on transcription and sentiment.
    """
    word_count = len(text.split())
    filler_count = count_filler_words(text)
    filler_ratio = filler_count / word_count if word_count > 0 else 0

    # Simple analysis: if more than 3% of words are fillers, suggest reduction.
    feedback_points = []

    if filler_ratio > 0.03:
        feedback_points.append(
            f"Your speech contains {filler_count} filler words (approx. {filler_ratio*100:.1f}% of total words). "
            "Try to reduce these to keep your message clear."
        )
    else:
        feedback_points.append("Good job! Your use of filler words is minimal.")

    # Provide sentiment-oriented feedback.
    if sentiment["label"].upper() == "NEGATIVE":
        feedback_points.append("The overall sentiment of your speech is negative. Consider a more positive tone for public speeches.")
    else:
        feedback_points.append("Your tone is positive, which is effective for engaging your audience.")

    # Analyze sentence length using spaCy (optional; this can give hints on pacing and clarity)
    doc = nlp(text)
    sentence_lengths = [len(sent.text.split()) for sent in doc.sents]
    if sentence_lengths:
        avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths)
        if avg_sentence_length > 20:
            feedback_points.append(
                f"Your sentences average {avg_sentence_length:.1f} words. Consider shorter sentences to improve clarity."
            )
        else:
            feedback_points.append(
                f"Your sentence length (avg. {avg_sentence_length:.1f} words) is concise and generally good for clarity."
            )

    return {
        "total_words": word_count,
        "filler_word_count": filler_count,
        "feedback": feedback_points
    }

@app.post("/feedback/")
async def analyze_and_feedback(file: UploadFile = File(...)):
    # Save uploaded audio to a temporary file (supports .mp3 and .webm)
    # Determine file extension from uploaded file
    ext = ".webm"
    print("Uploaded file name:", file.filename)
    print("Saved to temp file:", temp_file_path)
    if file.filename.lower().endswith(".webm"):
        ext = ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        temp_file_path = tmp.name
    print("Saved to temp file:", temp_file_path)
    # Step 1: Transcribe audio using Whisper
    result = whisper_model.transcribe(temp_file_path)
    text = result["text"]

    # Step 2: Basic text analysis using spaCy
    doc = nlp(text)
    keywords = [chunk.text for chunk in doc.noun_chunks]
    named_entities = [(ent.text, ent.label_) for ent in doc.ents]
    word_count = len(text.split())

    # Step 3: Sentiment analysis with BERT-based pipeline
    sentiment = sentiment_pipeline(text[:512])[0]  # Limit to 512 tokens if necessary

    # Step 4: Generate public speaking feedback based on heuristics
    public_speaking_feedback = generate_feedback(text, sentiment)

    return {
        "transcription": text,
        "word_count": word_count,
        "keywords": keywords,
        "named_entities": named_entities,
        "sentiment": {
            "label": sentiment["label"],
            "score": round(sentiment["score"], 3)
        },
        "public_speaking_feedback": public_speaking_feedback
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
