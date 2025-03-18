

from actor.actor import YOLOActorPhoto


actor = YOLOActorPhoto(
    data_folder='C:\\Projects\\uav\\gen-model\\data\\testdata\\images',
    annotation_folder="C:\\Projects\\uav\\gen-model\\data\\testdata\\annotations",
    img_res=(88, 88),
    tinyYOLO_model=None,
    mini_batch_size=4
)

actor.load_data()