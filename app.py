from pdfminer.high_level import extract_text

import spacy
import cython
import re
import pickle

from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from sqlalchemy.exc import SQLAlchemyError

import numpy as np
import faiss
# import pinecone
from sentence_transformers import SentenceTransformer

DATABASE_URL = "postgresql://anshulshukla@localhost:5432/ats"
engine = create_engine(DATABASE_URL)


# Load spaCy NLP model
nlp = spacy.load("en_core_web_sm")


# Load Sentence Transformer Model
model = SentenceTransformer("all-MiniLM-L6-v2")


SKILLS_DB = [
    "Python", "JavaScript", "React", "Node.js", "GO", "Python", "Java", "Machine Learning", "Django", "SQL", 
    "Kafka", "AWS", "Data Science", "Deep Learning", "TensorFlow", "Kubernetes"
]

EMAIL_REGEX = r"[a-zA-Z0-9+_.-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]+"
PHONE_REGEX = r"\b(?:\+?91[-.\s]?)?\d{10}\b|\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"
DEGREES = ["B.Tech", "M.Tech", "MBA", "Ph.D", "B.Sc", "M.Sc", "Bachelor", "Master"]


def extract_email(text):
    """Extract email from text"""
    match = re.findall(EMAIL_REGEX, text)
    return match[0] if match else None

def extract_phone(text):
    """Extract phone number from text"""
    match = re.findall(PHONE_REGEX, text)
    return match[0] if match else None

def extract_education(text):
    """Extract education details"""
    found_degrees = [deg for deg in DEGREES if deg.lower() in text.lower()]
    return list(set(found_degrees))  # Remove duplicates

def extract_name(text):
    """Extract candidate name using spaCy NER"""
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return None


def extract_skills(text):
    """Extract skills from text by matching against a predefined skill set"""
    extracted_skills = [skill for skill in SKILLS_DB if skill.lower() in text.lower()]
    return list(set(extracted_skills))  # Remove duplicates


def preprocess_text(text):
    """Clean and preprocess resume text"""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    return text.strip()


def extract_resume_text(pdf_path):
    return extract_text(pdf_path)


# resume_text = extract_resume_text("/Users/anshulshukla/Downloads/Anshul_Shukla_-_Associate_Director_-_Navi,_Ex-_Flipkart.pdf")
# resume_text = extract_resume_text("/Users/anshulshukla/Downloads/Mohit_Jain_-_Engineering_Manager.pdf")
resume_text = extract_resume_text("/Users/anshulshukla/Downloads/Pooja_Mishra_-_Recruitment_Specialist.pdf")


# Apply Preprocessing
clean_resume = preprocess_text(resume_text)
# print(clean_resume)    

# Apply Skill Extraction
extracted_skills = extract_skills(clean_resume)
# print("Extracted Skills:", extracted_skills)

# Apply Extraction
parsed_resume = {
    "name": extract_name(resume_text),
    "email": extract_email(resume_text),
    "phone": extract_phone(resume_text),
    "skills": extracted_skills,
    "education": extract_education(resume_text),
    "embedding": model.encode(resume_text).tolist()
}

# print(parsed_resume)


def insert_resume(data):
    try:
        data = {k.lower(): v for k, v in data.items()}  # Convert keys to lowercase
        
        # 🔹 Ensure embedding is stored as a list of floats
        if isinstance(data["embedding"], np.ndarray):
            data["embedding"] = data["embedding"].tolist()  # Convert NumPy array to Python list
        
        # print("Inserting Data:", data)  # Debugging statement
        
        """Insert parsed resume into PostgreSQL"""
        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO resumes (name, email, phone, skills, education, embedding)
                VALUES (:name, :email, :phone, :skills, :education, :embedding)
            """), data)
            
            conn.commit()  # Ensure changes are committed
        
        print("Data inserted successfully!")

    except Exception as e:
        print("Error inserting data:", e)


# insert_resume(parsed_resume)



def load_resumes_from_db():
    """Load resumes and embeddings from PostgreSQL, handling different stored formats."""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT id, name, email, phone, skills, education, embedding FROM resumes"))
            rows = result.fetchall()
        
        print("Fetched resumes from database!")  # Debugging statement
        
        resumes = []
        embeddings = []
        metadata = {}

        for row in rows:
            resume = {col.lower(): row[idx] for idx, col in enumerate(result.keys())}

            # 🔹 Handle Pickle (`bytea`) format
            if isinstance(resume["embedding"], bytes):
                resume["embedding"] = pickle.loads(resume["embedding"])  # Convert bytes to float list

            # 🔹 Handle text format (if stored as a string)
            elif isinstance(resume["embedding"], str):
                try:
                    resume["embedding"] = eval(resume["embedding"])  # Convert string to list
                except:
                    resume["embedding"] = [float(x) for x in resume["embedding"].strip("{}").split(",")]

            # 🔹 Ensure embeddings are NumPy float32 arrays
            resumes.append(resume["name"])
            embeddings.append(np.array(resume["embedding"], dtype="float32"))
            metadata[resume["id"]] = resume  # Store full metadata for retrieval

        return resumes, np.array(embeddings), metadata

    except Exception as e:
        print("Error fetching resumes:", e)
        return [], [], {}
# Load Data
resume_names, resume_vectors, resume_metadata = load_resumes_from_db()


# Initialize FAISS Index
DIM = 384  # Embedding size
index = faiss.IndexFlatL2(DIM)
index.add(resume_vectors)

print(f"Loaded {len(resume_names)} resumes into FAISS.")


def search_candidates(job_desc, top_k=3):
    """Find top-k candidates from PostgreSQL using FAISS"""
    job_vector = np.array([model.encode(job_desc)], dtype="float32")
    distances, indices = index.search(job_vector, top_k)

    results = []
    for idx in indices[0]:
        candidate_id = list(resume_metadata.keys())[idx]
        candidate = resume_metadata[candidate_id]
        results.append({
            "name": candidate["name"],
            "email": candidate["email"],
            "phone": candidate["phone"],
            "skills": candidate["skills"],
            "education": candidate["education"]

        })
    
    return results

# Example Query
job_description = "Looking for Navi"
matches = search_candidates(job_description, top_k=2)

print("\nTop Matches:")
for match in matches:
    print(match)



