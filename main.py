import argparse

from src.trainer import Trainer, Hyper_tuner


def str2bool(v):
    return v.lower() in ("true")


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--train", action="store_true", default=False)
    
    #parser.add_argument("--train", action="store_true", default=True)
    #parser.add_argument("--hyper", action="store_true", default=False)

    parser.add_argument("--epoch", type=int, default=3, help="The number of epochs to run")
    parser.add_argument("--batch_size", type=int, default=16, help="The size of batch per gpu")
    parser.add_argument("--print_freq", type=int, default=2, help="The number of image_print_freqy")
    parser.add_argument("--save_freq", type=int, default=50, help="The number of ckpt_save_freq")

    parser.add_argument("--g_opt", type=str, default="adam", help="learning rate for generator")
    parser.add_argument("--d_opt", type=str, default="adam", help="learning rate for discriminator")
    
    #parser.add_argument("--g_lr", type=float, default=0.0001, help="learning rate for generator")
    parser.add_argument("--g_lr", type=float, default=0.01, help="learning rate for generator")
    #parser.add_argument("--d_lr", type=float, default=0.0004, help="learning rate for discriminator")
    parser.add_argument("--d_lr", type=float, default=0.04, help="learning rate for discriminator")
    parser.add_argument("--beta1", type=float, default=0.0, help="beta1 for Adam optimizer")
    parser.add_argument("--beta2", type=float, default=0.9, help="beta2 for Adam optimizer")
    parser.add_argument("--gpl", type=float, default=10, help="The gradient penalty lambda")


    parser.add_argument("--z_dim", type=int, default=128, help="Dimension of noise vector")
    parser.add_argument("--image_size", type=int, default=128, help="The size of image")
    parser.add_argument("--sample_num", type=int, default=15, help="The number of sample images")

    parser.add_argument("--g_conv_filters", type=int, default=16, help="basic filter num for generator")
    parser.add_argument("--g_conv_kernel_size", type=int, default=4, help="basic kernel size for generator")
    parser.add_argument("--d_conv_filters", type=int, default=16, help="basic filter num for disciminator")
    parser.add_argument("--d_conv_kernel_size", type=int, default=4, help="basic kernel size for disciminator")

    parser.add_argument("--restore_model", action="store_true", default=False, help="the latest model weights")
    parser.add_argument("--g_pretrained_model", type=str, default=None, help="path of the pretrained model")
    parser.add_argument("--d_pretrained_model", type=str, default=None, help="path of the pretrained model")

    #parser.add_argument("--data_path", type=str, default="./data/train")
    parser.add_argument("--data_path", type=str, default="./data/train_semantic")

    


    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoint", help="Directory name to save the checkpoints"
    )
    parser.add_argument("--result_dir", type=str, default="results", help="Directory name to save the generated images")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory name to save training logs")
    parser.add_argument(
        "--sample_dir", type=str, default="samples", help="Directory name to save the samples on training"
    )
    parser.add_argument(
        "--experiment_name", type=str, default="/experiment_name", help="name of the experiment with timestamp"
    )

    

    parser.add_argument("--category_file", type=str, default="./resources/category.csv")

    return parser.parse_args()


def main():
    args = parse_args()
    trainer = Trainer(args)

    if args.train:
        trainer.train()   
    else:
        trainer.test()


# alex added this for hyper_tuning .sh
def tune():
    args = parse_args()
    trainer = Hyper_tuner(args)
    #print(args)
    

from  checkpoint.list_of_checkoints import return_checkpoints




# generate semantic images from pre-trained model:
def generate_from_model(args, g_checkpoint):
    


    # remove the last part of the checkpoint
    args.g_pretrained_model = g_checkpoint.replace(".index","")
        
    #args.d_pretrained_model = "models/checkpoints/d/cp-007000.ckpt"
    args.sample_num =35

    # get experminet name form the chekcp name
    slashes= args.g_pretrained_model.split("akin-generator/checkpoint/")[1].split("/")
    
    slashes[-1]= slashes[-1].replace(".ckpt","")

    args.experiment_name =slashes[0] + slashes[-1]
    print(args.experiment_name)
    args.result_dir = "/data1/data_alex/rico/akin_experiments/samples_generator/"

    trainer = Trainer(args)
    
    trainer.test()


   

def samples_generator(): 

    args = parse_args()

    #g_checkpoint= "/home/atsumilab/alex/rico/akin-generator/checkpoint/exp11 20230124-221114 lr 0.0001 0.0004 0.0 0.9 10/g/cp-007000.ckpt.index"

    
    g_checkpoint ="../akin-generator/checkpoint/pretrained checkpoints/g/cp-007000.ckpt.index"
    #args.g_pretrained_model ="/home/atsumilab/alex/rico/akin-generator/checkpoint/d7500 where we continued on pretrained/g/cp-007300.ckpt"
    
    generate_from_model(args,g_checkpoint)


    # incase of multple checkoiints

    chpts = return_checkpoints()

    for g_checkpoint in chpts:
        print (g_checkpoint)
        #generate_from_model(args,g_checkpoint)
     
        



if __name__ == "__main__":
    main()
    #tune()
    #samples_generator()
    
