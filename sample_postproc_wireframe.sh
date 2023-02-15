#!/bin/bash
for i in {0..10..2}
do
  echo "run test $i times"
done


"""
here i have to write a shell that


sampler


it's a separated command, prefered to be done alone in a separated file using main  

python main.py --g_pretrained_model models/checkpoints/g/cp-007000.ckpt --d_pretrained_model models/checkpoints/d/cp-007000.ckpt


output of this process will be save in a folder lett's say called 
'/data1/data_alex/rico/akin_experiments/samples_generator/model_name_chechkpoint7000'

so postproc: will be as :

python src/postProcessing.py --semantic_images_folder '/data1/data_alex/rico/akin_experiments/samples_generator/pretrained checkpointscp-007000' --destination_path 'data/postprocessed_pretraied'

this will generate images and put it in post process folder





generate wireframe images from json


python src/prototypeGenerator.py --font_path resources/fonts/DroidSans.ttf

python src/prototypeGenerator.py --json_file_location 'data/postprocessed_pretraied' --destination_folder 'data/final_pretrained'

and wireframe 

then inception and fid

"""

gen_samples="/data1/data_alex/rico/akin_experiments/samples_generator/badarch2_20230126-150905_lr_0.005_0.02_0.0_0.9_10cp-002650"

gen_samples="/home/atsumilab/alex/rico/Self-Attention-GAN/results/sagan_akin3"

postproc_path="/data1/data_alex/rico/akin_pytorch/postproc_sagan_akin3_13000"

final_path="/data1/data_alex/rico/akin_pytorch/final_sagan_akin3_13000"

python src/postProcessing.py --semantic_images_folder $gen_samples  --destination_path $postproc_path

python src/prototypeGenerator.py --json_file_location $postproc_path --destination_folder $final_path

python inception_train.py --data_dir $final_path

### last command
###  python main.py --train --g_lr 0.0002 --d_lr 0.0002 --restore_model