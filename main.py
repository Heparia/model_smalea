from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from tflite_runtime.interpreter import Interpreter
from fastapi.staticfiles import StaticFiles
import urllib.parse
from starlette.responses import Response
import numpy as np
import asyncio
import pandas as pd
from PIL import Image
import time
from io import BytesIO
import os

app = FastAPI()

class CORSStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope) -> Response:
        response = await super().get_response(path, scope)
        # Tambahkan header ini agar library PDF di frontend bisa membaca gambar
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
        return response
    
app.mount("/static", CORSStaticFiles(directory="static"), name="static")

origins = [
    "http://localhost:5173",
    "https://smalea-leaf-classifier.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,      
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

semaphore = asyncio.Semaphore(1)

class_names = ['bukan daun','ganja', 'jambu biji', 'jati blanda', 'jungrahab', 'katuk', 'keji beling', 'kemuning', 'koka', 'kumis kucing', 'saga', 'salam', 'seledri', 'sereh', 'sirih', 'sirsak', 'tapak liman', 'teh', 'tempuyung', 'urang aring']
IMG_SIZE = (224, 224)

# Load TFLite model
interpreter = Interpreter(model_path="./model_fiks.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter_feature = Interpreter(model_path="./data/feature_model.tflite")
interpreter_feature.allocate_tensors()

input_feature = interpreter_feature.get_input_details()
output_feature = interpreter_feature.get_output_details()

embeddings = np.load("./data/embedding.npy", mmap_mode="r") 
labels = np.load("./data/labels.npy", mmap_mode="r")
image_paths = np.load("./data/image_paths.npy")

def cosine_similarity(query, embeddings):
    norm = np.linalg.norm(query)
    if norm == 0:
        return np.zeros(len(embeddings))

    query_norm = query / norm
    sims = np.dot(embeddings, query_norm)
    return sims

def find_most_similar(query_embedding):
    sims = cosine_similarity(query_embedding, embeddings)
    idx = np.argmax(sims)

    return {
        "class": class_names[labels[idx]],
        "image_path": image_paths[idx],
        "score": float(sims[idx])
    }

def prepare_image(file_bytes: bytes) -> np.ndarray:
    img = Image.open(BytesIO(file_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img)
    img_batch = np.expand_dims(img_array, axis=0).astype(np.float32) 
    return img_batch

df = pd.read_csv("data.csv")
df["jenis_daun_normalized"] = df["jenis_daun"].str.strip().str.lower()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:

        contents = await file.read()
        input_data = prepare_image(contents)

        # QUEUE INFERENCE
        async with semaphore:

            interpreter.set_tensor(input_details[0]['index'], input_data)

            start_time = time.perf_counter()

            interpreter.invoke()

            end_time = time.perf_counter()
            inference_time_ms = (end_time - start_time) * 1000

            output_index = output_details[0]["index"]
            predictions = interpreter.get_tensor(output_index)[0]

        predictions = np.nan_to_num(predictions, nan=0.0, posinf=1.0, neginf=0.0)

        all_idx = predictions.argsort()[::-1]

        results_1 = [
            {
                "label": df.iloc[idx]["jenis_daun"],
                "confidence": float(predictions[idx]),
            }
            for idx in all_idx
        ]

        results_2 = []

        for idx in all_idx:
            label = class_names[idx]

            row = df[df["jenis_daun"] == label].iloc[0]

            results_2.append(
                row.replace([np.nan, np.inf, -np.inf], "").to_dict()
            )

        return JSONResponse(
            content={
                "results": results_1,
                "detail": results_2,
                "inference_time_ms": inference_time_ms
            }
        )

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/embedding")
async def embedding(file: UploadFile = File(...), top_k: int = 5):
    sims = None
    pred_shape = None
    pred_lain = None
    try:
        contents = await file.read()
        input_data = prepare_image(contents)
        async with semaphore:
            interpreter.set_tensor(input_details[0]['index'], input_data)

            start_time = time.perf_counter()

            interpreter.invoke()
            output_index = output_details[0]["index"]
            predictions = interpreter.get_tensor(output_index)[0]

            predictions = np.nan_to_num(predictions, nan=0.0, posinf=1.0, neginf=0.0)

            top_idx = int(np.argmax(predictions))
            top_conf = float(predictions[top_idx])
            top_label = class_names[top_idx]

            interpreter_feature.set_tensor(input_feature[0]['index'], input_data)
            interpreter_feature.invoke()

            pred = interpreter_feature.get_tensor(output_feature[0]['index'])
            pred = np.squeeze(pred)
            end_time = time.perf_counter()
            inference_time_ms = (end_time - start_time) * 1000
        

        all_idx = predictions.argsort()[::-1]

        results_1 = [
            {
                "label": df.iloc[idx]["jenis_daun"],
                "confidence": float(predictions[idx]),
            }
            for idx in all_idx
        ]

        results_2 = []

        for idx in all_idx:
            label = class_names[idx]

            row = df[df["jenis_daun"] == label].iloc[0]

            results_2.append(
                row.replace([np.nan, np.inf, -np.inf], "").to_dict()
            )

        sims = cosine_similarity(pred, embeddings)
        sorted_indices = np.argsort(sims)[::-1]

        results = []
        anchor_found = False

        for i, idx in enumerate(sorted_indices):
            path = str(image_paths[idx])
            path_baru = path.replace("train/train/", "static/")
            safe_url = urllib.parse.quote(path_baru)

            # ambil label dari path
            path_label = path.split("/")[2]  # atau pakai os.path

            if not anchor_found:
                if path_label == top_label:
                    # ini anchor (hasil pertama sesuai klasifikasi)
                    anchor_found = True
                    start_index = i

                    results.append({
                        "label": str(labels[idx]),
                        "similarity": float(sims[idx]),
                        "path": safe_url
                    })
            else:
                # setelah anchor → bebas ambil
                results.append({
                    "label": str(labels[idx]),
                    "similarity": float(sims[idx]),
                    "path": safe_url
                })

            if len(results) >= top_k:
                break
        return {
            "results": results,
            "top classification": top_label,
            "top confidence": top_conf,
            "interface time": inference_time_ms
        }
        # return result

    except Exception as e:
        return {
            "error": str(e),
            "pred_shape": pred_shape,
            "pred lain": pred_lain,
            "sims_type": str(type(sims))
        }

# Endpoint root (informasi dasar API)
@app.get("/")
def root():
    return {"message": "API Daun Herbal - Gunakan POST /predict untuk klasifikasi gambar daun.", "data_daun": df.to_dict()}

@app.get("/library")
async def get_all_daun():
    try:
        data = (
            df.replace([np.nan, np.inf, -np.inf], "")
            .to_dict(orient="records")
        )

        return {"total": len(data), "data": data}

    except Exception as e:
        return {"error": str(e)}

@app.get("/library/{nama_daun}")
async def get_daun(nama_daun: str):
    try:
        nama_daun = nama_daun.strip().lower()
        row = df[df["jenis_daun_normalized"] == nama_daun].iloc[0]
        return {"result": row.drop("jenis_daun_normalized").to_dict()}
    except IndexError:
        return {"error": "Daun tidak ditemukan"}

