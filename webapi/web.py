from fastapi import FastAPI, File, UploadFile, HTTPException, Form
import numpy as np
from io import BytesIO
from ..facedectapi import DeepFaceDect
from pydantic import BaseModel
import base64
import cv2

app = FastAPI()


class Params(BaseModel):
    user_name: str
    face_in_name: str


@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
async def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}

@app.post("/upload_pic")
async def upload_pic_from_user(file: UploadFile = File(...),
                               params:  Params = Form(...)
                               ):
    try:
        # 读取上传的文件内容
        image_bytes = await file.read()

        # 使用 BytesIO 来创建一个可读的对象
        image_stream = BytesIO(image_bytes)

        # 使用 OpenCV 读取图像
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

        if image is not None:
            faces = DeepFaceDect.get_all_face_in_pic(image)
            if len(faces) >= 2:
                return {
                    "message" : "picture has two face"
                }
        if faces is not None:
            DeepFaceDect.upload_face_from_user(image, params.user_name, params.face_in_name)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/facedect/facesDect")
async def dect_all_face(file: UploadFile = File(...),
                        params: Params = Form(...)
                        ):
    try:
