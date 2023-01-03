from flask import Blueprint, render_template, url_for, request
from gradcam_model import *
import os
from models.model import Image
import datetime
from main import db

bp = Blueprint('upload', __name__, url_prefix='/')


@bp.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    filename = file.filename
    file.save(os.path.join('static/inputs', filename))
    img_src = url_for('static', filename='inputs/' + filename)

    label, prob, heatmap_path, gb_path, cam_gb_path = result(img_src)

    gb_path = url_for('static', filename='outputs/' + "gb.jpg")
    cam_gb_path = url_for('static', filename='outputs/' + "Guided_Grad_CAM.jpg")
    heatmap_path = url_for('static', filename='outputs/' + "Heat_Map.jpg")

    image = Image(input_image_file_name=filename,
                  input_image_file_path=img_src,
                  heatmap_file_path=heatmap_path,
                  gb_file_path=gb_path,
                  cam_gb_file_path=cam_gb_path,
                  prob=prob,
                  label=label,
                  created_date=datetime.datetime.now(),
                  )

    db.session.add(image)
    db.session.commit()
    return render_template('index.html', filename=img_src, label=label, probability=prob, heatmap=heatmap_path, gb=gb_path, cam_gb=cam_gb_path)