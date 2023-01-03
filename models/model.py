from flask_sqlalchemy import SQLAlchemy


db = SQLAlchemy()


class Image(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    input_image_file_name = db.Column(db.String(256))
    input_image_file_path = db.Column(db.String(256))
    heatmap_file_path = db.Column(db.String(256))
    gb_file_path = db.Column(db.String(256))
    cam_gb_file_path = db.Column(db.String(256))
    prob = db.Column(db.String(50))
    label = db.Column(db.String(50))
    created_date = db.Column(db.DateTime(), nullable=False)

    def __init__(self, input_image_file_name, input_image_file_path,
                 heatmap_file_path, gb_file_path, cam_gb_file_path,
                 prob, label, created_date):
        self.input_image_file_name = input_image_file_name
        self.input_image_file_path = input_image_file_path
        self.heatmap_file_path = heatmap_file_path
        self.gb_file_path = gb_file_path
        self.cam_gb_file_path = cam_gb_file_path
        self.prob = prob
        self.label = label
        self.created_date = created_date

    @property
    def serialize(self):
        return {
            'id': self.id,
            'input_image_file_name': self.input_image_file_name,
            'input_image_file_path': self.input_image_file_path,
            'heatmap_file_path': self.heatmap_file_path,
            'gb_file_path': self.gb_file_path,
            'cam_gb_file_path': self.cam_gb_file_path,
            'prob': self.prob,
            'label': self.pred_class,
            'created_date': self.created_date
        }