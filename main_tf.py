
import platform
from cnn.pkg.models.tf_tinysimmoYOLO import TFTinysimmoYOLOModel
from actor.tf_actor import TFYOLOActorPhoto


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


model = TFTinysimmoYOLOModel(model_path='./model', S=4, B=2, C=11)
actor = TFYOLOActorPhoto(
    data_folder=data_folder,
    annotation_folder=annotation_folder,
    model_input_img_res=(88, 88),
    model=model,
    mini_batch_size=10
)


def test_model():
    actor.model.create_new_model()
    actor.train_model(loops=1)
    actor.print_results(image_path=data_folder + '/0000001_02999_d_0000005.jpg')


test_model()
