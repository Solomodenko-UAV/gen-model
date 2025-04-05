
import platform

import tensorflow as tf
from cnn.pkg.models.tf_tinysimmoYOLO import TFTinysimmoYOLOModel
from actor.tf_actor import TFYOLOActorPhoto


data_folder = ''
annotation_folder = ''
if platform.system() == "Windows":
    data_folder = 'C:\\Projects\\uav\\real_data\\VisDrone2019-DET-test-dev\\images'
    annotation_folder = "C:\\Projects\\uav\\real_data\\VisDrone2019-DET-test-dev\\annotations"
    # data_folder = 'C:\\Projects\\uav\\real_data\\VisDrone2019-DET-train\\images'
    # annotation_folder = "C:\\Projects\\uav\\real_data\\VisDrone2019-DET-train\\labels"
elif platform.system() == "Darwin":
    # data_folder = '/Users/kana/Projects/my_own/pet_projects/VisDrone2019-DET-train/images'
    # annotation_folder = "/Users/kana/Projects/my_own/pet_projects/VisDrone2019-DET-train/labels"
    data_folder = '/Users/kana/Projects/my_own/pet_projects/gen-model/data/VisDrone2019-DET-val/testdata/images'
    annotation_folder = "/Users/kana/Projects/my_own/pet_projects/gen-model/data/VisDrone2019-DET-val/testdata/annotations"
else:
    print("Running on an unsupported OS")


model = TFTinysimmoYOLOModel(model_path='./model', S=80, B=5, C=11)
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
    history = actor.train_model(loops=1100)
    actor.plot_training_history(history)
    actor.print_results(image_path=data_folder + '/0000006_00159_d_0000001.jpg')


test_model()
