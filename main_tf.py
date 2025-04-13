
import platform

import tensorflow as tf
from cnn.pkg.layers.yolo_output.tf_layer import TFYoloOutput
from cnn.pkg.models.tf_tinysimmoYOLO import TFTinysimmoYOLOModel
from actor.tf_actor import TFYOLOActorPhoto


data_folder = ''
annotation_folder = ''
if platform.system() == "Windows":
    data_folder = 'C:\\Projects\\uav\\real_data\\VisDrone2019-DET-train\\images'
    annotation_folder = "C:\\Projects\\uav\\real_data\\VisDrone2019-DET-train\\labels"
    # data_folder = 'C:\\Projects\\uav\\real_data\\VisDrone2019-DET-train\\images'
    # annotation_folder = "C:\\Projects\\uav\\real_data\\VisDrone2019-DET-train\\labels"
elif platform.system() == "Darwin":
    # data_folder = '/Users/kana/Projects/my_own/pet_projects/VisDrone2019-DET-train/images'
    # annotation_folder = "/Users/kana/Projects/my_own/pet_projects/VisDrone2019-DET-train/labels"
    data_folder = '/Users/kana/Projects/my_own/pet_projects/gen-model/data/VisDrone2019-DET-val/testdata/images'
    annotation_folder = "/Users/kana/Projects/my_own/pet_projects/gen-model/data/VisDrone2019-DET-val/testdata/annotations"
elif platform.system() == "Linux":
    data_folder = '/mnt/c/Projects/uav/real_data/VisDrone2019-DET-train/images'
    annotation_folder = "/mnt/c/Projects/uav/real_data/VisDrone2019-DET-train/labels"
else:
    print(platform.system())
    print("Running on an unsupported OS")

S = 11
B = 2
C = 11
model = TFTinysimmoYOLOModel(S, B, C)
actor = TFYOLOActorPhoto(
    data_folder=data_folder,
    annotation_folder=annotation_folder,
    model_input_img_res=(88, 88),
    model=model,
    mini_batch_size=10
)

def test_model():
    # tf.config.run_functions_eagerly(True)
    
    actor.model.create_new_model()
    # loops = 200
    loops = 1
    batch_size = 16
    history = actor.train_model(loops, batch_size)
    actor.model.save_model(folder_path='./trained_models', model_name=f"tinysimmo_yolo_loops={loops}_batch_size={batch_size}_B={B}_C={C}_S={S}")
    actor.plot_training_history(history)
    # actor.model.load_model(folder_path='./model', model_name='tinysimmo_yolo')
    actor.print_results(
        image_path=data_folder + '/0000002_00005_d_0000014.jpg',
        score_threshold=0.5,
        iou_threshold=0.5,
    )
    
def debug():
    pass
    

# debug()
test_model()
