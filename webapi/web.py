from fastapi import FastAPI, File, UploadFile, HTTPException, Form
import numpy as np
from io import BytesIO
from fastapi.responses import FileResponse


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
                               params: Params = Form(...)
                               ):
    try:
        # 读取上传的文件内容
        image_bytes = await file.read()

        # 使用 OpenCV 读取图像
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        # 将bgr转化为rgb
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image_rgb is not None:
            faces = DeepFaceDect.get_all_face_in_pic(image_rgb)
        if faces is not None:
            if len(faces) >= 2:
                return {
                    "message": "picture has two face"
                }
            else:
                DeepFaceDect.upload_face_from_user(image_rgb, params.user_name, params.face_in_name)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/facedect/facesDect")
async def dect_all_face(file: UploadFile = File(...),
                        params: Params = Form(...)
                        ):
    try:
        image_bytes = await file.read()
        image_bgr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        faces = DeepFaceDect.get_all_face_in_pic(image_rgb)
        if faces is not None:
            for face in faces:
                compare_result = DeepFaceDect.compare_and_verify_face(face['face'], params.user_name)
                point_at_rightDown=(face['facial_area']['x'] + face['facial_area']['w'], face['facial_area']['y'] + face['facial_area']['h'])
                cv2.rectangle(image_rgb, (face['facial_area']['x'], face['facial_area']['y']),
                              point_at_rightDown,
                              (0, 0, 255),
                              2)
                if compare_result != 'not found match face':
                    cv2.putText(image_rgb,
                                compare_result,
                                (face['facial_area']['x'], face['facial_area']['y']),
                                cv2.FONT_HERSHEY_PLAIN,
                                0.7,
                                (0, 0, 255)
                                )
            _, buffer = cv2.imencode('.jpg', image_bgr)
            image_bytes_return = buffer.tobytes()
            return FileResponse(BytesIO(image_bytes_return), filename="face_dect_image.jpg", media_type="image/jpeg")

            return

        else:
            return {
                "message": "no face"
            }

    except Exception as e:
        # 如果发生错误，返回错误信息
        raise HTTPException(status_code=500, detail=str(e))
