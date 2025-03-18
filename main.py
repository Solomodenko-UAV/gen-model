

from actor.actor import YOLOActorPhoto


actor = YOLOActorPhoto(
    data_folder="/Users/kana/Projects/my_own/pet_projects/gen-model/data/VisDrone2019-DET-val/test_data_folder",
    annotation_folder="annotations",
    img_res=(88, 88),
    tinyYOLO_model=None,
    mini_batch_size=4
)

actor.load_data()