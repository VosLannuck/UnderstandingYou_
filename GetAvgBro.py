import numpy as np
vanilla_alex_acc = [
    0.6778,
    0.6667,
    0.6667,
    0.6611,
    0.6611,
    0.6556,
    0.6556,
    0.65,
    0.6444,
    0.6444,
    0.6444,
    0.6278,
    0.6222,
    0.6167,
    0.6111,
    0.6056,
    0.6,
    0.5833,
    0.5833,
    0.5333,
    0.5167,
    0.5167,
    0.5111,
    0.5111,
    0.5056,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5
]

pretrained_alex_acc = [
    0.6167,
    0.6667,
    0.55,
    0.6889,
    0.7167,
    0.7056,
    0.6333,
    0.6722,
    0.5667,
    0.5944,
    0.7111,
    0.6667,
    0.7389,
    0.6889,
    0.6667,
    0.6944,
    0.7056,
    0.6833,
    0.6333,
    0.65,
    0.6833,
    0.6778,
    0.7,
    0.7222,
    0.6333,
    0.6667,
    0.55,
    0.7111,
    0.6556,
    0.6556,
]

vanilla_resnet_acc = [
    0.6944,
    0.5944,
    0.6889,
    0.6167,
    0.7,
    0.6167,
    0.7167,
    0.6778,
    0.7167,
    0.6944,
    0.6944,
    0.6611,
    0.7167,
    0.7167,
    0.6667,
    0.6556,
    0.6389,
    0.6722,
    0.6944,
    0.7056,
    0.7056,
    0.6111,
    0.6389,
    0.6889,
    0.6722,
    0.6778,
    0.6722,
    0.7278,
    0.65,
    0.6833,
]

pretrained_resnet_acc = [
    0.8389,
    0.7444,
    0.7722,
    0.7833,
    0.7778,
    0.7556,
    0.7,
    0.8167,
    0.7889,
    0.7778,
    0.7833,
    0.7778,
    0.6944,
    0.7889,
    0.7833,
    0.8222,
    0.6611,
    0.7889,
    0.7222,
    0.5611,
    0.7944,
    0.6111,
    0.6167,
    0.8,
    0.7444,
    0.6889,
    0.7889,
    0.7167,
    0.8,
    0.8167,
]

vanilla_vgg_acc = [
    0.6778,
    0.6333,
    0.6278,
    0.65,
    0.6444,
    0.5611,
    0.5889,
    0.6833,
    0.6278,
    0.6667,
    0.5944,
    0.6944,
    0.5,
    0.6278,
    0.6611,
    0.6222,
    0.6389,
    0.6389,
    0.5111,
    0.7333,
    0.65,
    0.5,
    0.5,
    0.6944,
    0.5056,
    0.6389,
    0.6667,
    0.6389,
    0.7611,
    0.7,
]

pretrained_vgg16_acc = [
    0.7222,
    0.6778,
    0.7056,
    0.6667,
    0.6722,
    0.6611,
    0.7111,
    0.7333,
    0.6556,
    0.6833,
    0.6611,
    0.6056,
    0.7222,
    0.7222,
    0.6944,
    0.65,
    0.7278,
    0.6333,
    0.7167,
    0.6389,
    0.6278,
    0.7,
    0.7389,
    0.7,
    0.6889,
    0.7167,
    0.7167,
    0.6667,
    0.6556,
    0.6389,
]

vit_acc = [
    0.9167,
    0.8722,
    0.8611,
    0.85,
    0.8444,
    0.8389,
    0.8111,
    0.8,
    0.7889,
    0.7833,
    0.7722,
    0.7667,
    0.7,
    0.6389,
    0.6167,
    0.5889,
    0.5889,
    0.5111,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
]

if __name__ == "__main__":
    npVanillaAlex = np.array(vanilla_alex_acc)
    npVanillaVGG = np.array(vanilla_vgg_acc)
    npVanillaResnet = np.array(vanilla_resnet_acc)

    npPretrainedAlex = np.array(pretrained_alex_acc)
    npPretrainedVGG = np.array(pretrained_vgg16_acc)
    npPretrainedResnet = np.array(pretrained_resnet_acc)

    npVIT = np.array(vit_acc)

    print(f"Len Alex: {len(npVanillaAlex)}")
    print(f"Len VGG: {len(npVanillaVGG)}")
    print(f"Len Resnet: {len(npVanillaResnet)}")
    print(f"Len PreAlex: {len(npPretrainedAlex)}")
    print(f"Len PreVGG: {len(npPretrainedVGG)}")
    print(f"Len PreResnet: {len(npPretrainedResnet)}")
    print(f"Len VIT: {len(npVIT)}")
    
    print("=" * 30)
    print("=" * 30)
   
    print(f"Average Accuracy Vanilla Alex: {np.average(npVanillaAlex)}")
    print(f"Average Accuracy Vanilla VGG: {np.average(npVanillaVGG)}")
    print(f"Average Accuracy Vanilla Resnet: {np.average(npVanillaResnet)}")
    print(f"Average Accuracy pretrained Alex: {np.average(npPretrainedAlex)}")
    print(f"Average Accuracy pretrained VGG: {np.average(npPretrainedVGG)}")
    print(f"Average Accuracy pretrained Resnet: {np.average(npPretrainedResnet)}")
    print(f"Average Accuracy VIT: {np.average(npVIT)}")
    
    print("=" * 30)
    print("=" * 30)
    
    print(f"Best Accuracy Vanilla Alex: {np.max(npVanillaAlex)}")
    print(f"Best Accuracy Vanilla VGG: {np.max(npVanillaVGG)}")
    print(f"Best Accuracy Vanilla Resnet: {np.max(npVanillaResnet)}")
    print(f"Best Accuracy pretrained Alex: {np.max(npPretrainedAlex)}")
    print(f"Best Accuracy pretrained VGG: {np.max(npPretrainedVGG)}")
    print(f"Best Accuracy pretrained Resnet: {np.max(npPretrainedResnet)}")
    print(f"Best Accuracy VIT: {np.max(npVIT)}")

