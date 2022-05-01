docker run \
        --volume=/media/hd03/axot_data/image_title_match_submit/submit:/workspace/image_title_match \
        --volume=/media/md01/home/axot/img_text_data/data:/workspace/image_title_match/contest_data \
        --shm-size=10g \
        --gpus=all \
        --name=submit \
        -it img_text:v1 \
        bash
