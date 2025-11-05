import time
from train import train
from Modules import generate
from args import args
import utils
import torch
import os
from Net import Generator_DWT

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# flag = 1
flag = 0

if flag == 1:
    IS_TRAINING = True
else:
    IS_TRAINING = False


def load_model(model_path):
    G_model = Generator_DWT()
    G_model.load_state_dict(torch.load(model_path), False)
    print('# generator parameters:', sum(param.numel() for param in G_model.parameters()))
    G_model.eval()
    G_model.cuda()
    return G_model


def main():
    # training
    if IS_TRAINING:
        train_data_ir = utils.list_images(args.train_ir)
        train_data_vi = utils.list_images(args.train_vi)
        train(train_data_ir, train_data_vi)


    else:
        print("\nBegin to generate pictures ...\n")

        #test_imgs_path_ir = "./test_imgs/Harvard/FDG/"
        #test_imgs_path_vi = "./test_imgs/Harvard/T1/"
        #test_imgs_path_vi = "./test_imgs/MSRS/test/vi/"
        #test_imgs_path_ir = "./test_imgs/MSRS/test/ir/"
        #test_imgs_path_vi = "./test_imgs/TNO/vi/"
        #test_imgs_path_ir = "./test_imgs/TNO/ir/"
        #test_imgs_path_vi = "./test_imgs/VIS-SAR-1000-0.8/vi/train/"
        #test_imgs_path_ir = "./test_imgs/VIS-SAR-1000-0.8/ir/train/"
        #test_imgs_path_vi = "./test_imgs/awm/vi2/"
        #test_imgs_path_ir = "./test_imgs/awm/ir/"
        #test_imgs_path_vi = "./test_imgs/snow/vi/"
        #test_imgs_path_ir = "./test_imgs/snow/ir/"
        #test_imgs_path_vi = "./test_imgs/3/vi/"
        #test_imgs_path_ir = "./test_imgs/3/ir/"
        #test_imgs_path_vi = "./test_imgs/4/ir/"
        #test_imgs_path_ir = "./test_imgs/4/vi/"
        #test_imgs_path_vi = "./test_imgs/5_M3FD/ir/"
        #test_imgs_path_ir = "./test_imgs/5_M3FD/vi/"
        #test_imgs_path_vi = "./test_imgs/M3FD/vi/"
        #test_imgs_path_ir = "./test_imgs/M3FD/ir/"
        #test_imgs_path_vi = "./test_imgs/4_LLVIP/vi/"
        #test_imgs_path_ir = "./test_imgs/4_LLVIP/ir/"
        #test_imgs_path_vi = "./test_imgs/3_road/VI_RGB/"
        #test_imgs_path_ir = "./test_imgs/3_road/ir/
        test_imgs_path_vi = r"C:\Users\PC\Desktop\com\data\TNO\vi2"
        test_imgs_path_ir = r"C:\Users\PC\Desktop\com\data\TNO\ir2"


        test_imgs_path_fir = "./test_imgs/MSRS/fir/"
        vi_files = sorted(os.listdir(test_imgs_path_vi))
        model_49 = load_model(os.path.join(os.getcwd(), 'FusionModel', 'Final_G_Epoch_99.model'))
        model_49.eval()
        model_49.cuda()
        model_59 = load_model(os.path.join(os.getcwd(), 'FusionModel', 'Final_G_Epoch_99.model'))
        model_59.eval()
        model_59.cuda()

        with torch.no_grad():
            for filename in vi_files:
                vi_path = os.path.join(test_imgs_path_vi, filename)
                ir_path = os.path.join(test_imgs_path_ir, filename)
                fir_path = os.path.join(test_imgs_path_fir, filename)
                index = int(filename[:5]) if filename[:5].isdigit() else filename

                if os.path.exists(ir_path):
                    # 用真实红外，模型49
                    print(f"Fusing {filename} with IR using model 49...")
                    generate(model_49, ir_path, vi_path, "results", index, mode='RGB')
                elif os.path.exists(fir_path):
                    # 用伪红外，模型59
                    print(f"Fusing {filename} with FIR using model 59...")
                    generate(model_59, fir_path, vi_path, "results", index, mode='RGB')
                else:
                    print(f"缺少红外或伪红外图像: {filename}")


if __name__ == "__main__":
    main()
