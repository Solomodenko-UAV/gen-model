

import time
from actor.actor import YOLOActorPhoto
from nn.pkg.models.tinysimmoYOLO import TinysimmoYOLOModel
import platform


# model = TinysimmoYOLOModel(model_path='model', S=4, B=2, C=1)
model = TinysimmoYOLOModel(model_path='./model', S=4, B=2, C=10)  # TODO: rollback

model.create_new_model(
    image_size=(88, 88),
    conv_l2_lambda=0.0001,
    fc_l2_lambda=0.0001,
    clip_value=5.0,
    conv_momentum=0.8,
)

if platform.system() == "Windows":
    data_folder = 'C:\\Projects\\uav\\gen-model\\data\\testdata\\images'
    annotation_folder = "C:\\Projects\\uav\\gen-model\\data\\testdata\\annotations"
elif platform.system() == "Darwin":
    data_folder = '/Users/kana/Projects/my_own/pet_projects/gen-model/data/VisDrone2019-DET-val/testdata/images'
    annotation_folder = "/Users/kana/Projects/my_own/pet_projects/gen-model/data/VisDrone2019-DET-val/testdata/annotations"
else:
    print("Running on an unsupported OS")

actor = YOLOActorPhoto(
    data_folder=data_folder,
    annotation_folder=annotation_folder,
    model_input_img_res=(88, 88),
    model=model,
    mini_batch_size=6
)

def train():
    start_time = time.time()
    actor.run_training_loop(epochs=1, learning_rate=0.001)
    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds")
    model.save_model()
    
def test():
    actor.func_for_tests(
        # show_model_boxes=True,
        evaluate=True,
        )
    
train()
    

