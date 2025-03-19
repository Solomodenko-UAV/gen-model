

from actor.actor import YOLOActorPhoto
from nn.pkg.models.tinysimmoYOLO import TinysimmoYOLOModel


# model = TinysimmoYOLOModel(model_path='model', S=4, B=2, C=1)
model = TinysimmoYOLOModel(model_path='model', S=4, B=2, C=10)  # TODO: rollback

model.create_new_model((88, 88))

actor = YOLOActorPhoto(
    data_folder='/Users/kana/Projects/my_own/pet_projects/gen-model/data/VisDrone2019-DET-val/testdata/images',
    annotation_folder="/Users/kana/Projects/my_own/pet_projects/gen-model/data/VisDrone2019-DET-val/testdata/annotations",
    model_input_img_res=(88, 88),
    model=model,
    mini_batch_size=4
)

actor.load_data()
