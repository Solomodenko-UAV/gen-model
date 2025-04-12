
import platform

import keras
import tensorflow as tf
from cnn.pkg.layers.yolo_output.tf_layer import TFYoloOutput
from cnn.pkg.models.tf_tinysimmoYOLO import TFTinysimmoYOLOModel
from actor.tf_actor import TFYOLOActorPhoto


data_folder = ''
annotation_folder = ''
if platform.system() == "Windows":
    data_folder = 'C:\\Projects\\uav\\real_data\\VisDrone2019-DET-train\\images'
    annotation_folder = "C:\\Projects\\uav\\real_data\\VisDrone2019-DET-train\\annotations"
    # data_folder = 'C:\\Projects\\uav\\real_data\\VisDrone2019-DET-train\\images'
    # annotation_folder = "C:\\Projects\\uav\\real_data\\VisDrone2019-DET-train\\labels"
elif platform.system() == "Darwin":
    # data_folder = '/Users/kana/Projects/my_own/pet_projects/VisDrone2019-DET-train/images'
    # annotation_folder = "/Users/kana/Projects/my_own/pet_projects/VisDrone2019-DET-train/labels"
    data_folder = '/Users/kana/Projects/my_own/pet_projects/gen-model/data/VisDrone2019-DET-val/testdata/images'
    annotation_folder = "/Users/kana/Projects/my_own/pet_projects/gen-model/data/VisDrone2019-DET-val/testdata/annotations"
else:
    print("Running on an unsupported OS")


model = TFTinysimmoYOLOModel(S=4, B=2, C=11)
actor = TFYOLOActorPhoto(
    data_folder=data_folder,
    annotation_folder=annotation_folder,
    model_input_img_res=(88, 88),
    model=model,
    mini_batch_size=10
)


def test_model():
    # tf.config.run_functions_eagerly(True)
    
    # actor.model.create_new_model()
    # history = actor.train_model(loops=300, batch_size=2)
    # actor.model.save_model(folder_path='./model', model_name='tinysimmo_yolo')
    # actor.plot_training_history(history)
    actor.model.load_model(folder_path='./model', model_name='tinysimmo_yolo')
    actor.print_results(
        image_path=data_folder + '/0000001_02999_d_0000005.jpg',
        score_threshold=0.5,
        iou_threshold=0.5,
    )
    
def debug():
    # layer = TFYoloOutput(
    #     S=4,
    #     B=2,
    #     C=11,
    #     anchors=tf.convert_to_tensor([0.5, 0.5]),
    # )
    
    # keras.models.save_model(
    #     layer,
    #     'model_test.keras',
    #     overwrite=True,
    # )
    
    # keras.models.load_model(
    #     'model_test.keras',
    #     custom_objects={
    #         'TFYoloOutput': TFYoloOutput,
    #     }
    # )
    
    model.save_model(
        folder_path='./model',
        model_name='tinysimmo_yolo',
    )
    model.load_mzodel(
        folder_path='./model',
        model_name='tinysimmo_yolo',
    )
    

# debug()
test_model()
