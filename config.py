'''
Author: yanhuo 1760331284@qq.com
Date: 2023-11-11 16:01:37
LastEditors: yanhuo 1760331284@qq.com
LastEditTime: 2024-03-08 17:04:33
FilePath: \text-gcd\config.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# -----------------
# DATASET ROOTS
# # -----------------
cifar_10_root = '/db/pszzz/NCD_dataset/cifar10'
cifar_100_root = '/db/pszzz/NCD_dataset/cifar100'
# cifar_100_root = '/wang_hp/zhy/data'
# cifar_10_root = '/db/psawl'
# cifar_100_root = '/db/psawl'
# cifar_10_root = '/home/zhun.zhong/GCD_dataset/cifar_10'
# cifar_100_root = '/home/zhun.zhong/GCD_dataset/cifar_100'

cub_root = '/db/pszzz/NCD_dataset/cub'
# cub_root = '/wang_hp/zhy/data'
# cub_root = '/db/psawl/cub'
# cub_root = '/home/zhun.zhong/GCD_dataset/cub'

aircraft_root = '/db/pszzz/NCD_dataset/aircraft/fgvc-aircraft-2013b'
# aircraft_root = '/home/zhun.zhong/GCD_dataset/fgvc_aircraft/fgvc-aircraft-2013b'

car_root = '/db/pszzz/NCD_dataset'
# car_root = '/wang_hp/zhy/data'
# car_root = '/home/zhun.zhong/GCD_dataset'

herbarium_dataroot = '/db/pszzz/NCD_dataset/herbarium_19'

imagenet_root = '/db/pszzz/NCD_dataset/imagenet'
# imagenet_root = '/mhug-storage/imagenet'

# oxford_pet_root = '/home/zhun.zhong/GCD_dataset'
# oxford_pet_root = '/db/pszzz/NCD_dataset'
oxford_pet_root = '/leonardo_work/IscrC_Fed-GCD/GCD_datasets'

# oxford_flowers_root = '/home/zhun.zhong/GCD_dataset'
oxford_flowers_root = '/db/pszzz/NCD_dataset'

# food_101_root = '/home/zhun.zhong/GCD_dataset/food_101'
food_101_root = '/db/pszzz/NCD_dataset/food-101'

# OSR Split dir
osr_split_dir = 'data/ssb_splits'

# tag root
cub_tag_root = '/home/pszzz/hyzheng/text-gcd/tag/cub_tags_attributes.txt'
aircraft_tag_root = '/home/pszzz/hyzheng/text-gcd/tag/aircraft_tags_attributes.txt'
car_tag_root = '/home/pszzz/hyzheng/text-gcd/tag/scars_tags_attributes.txt'
cifar_10_tag_root = 'tag/cifar10_tags_attributes.txt'
cifar_100_tag_root = '/home/pszzz/hyzheng/text-gcd/tag/cifar100_tags_attributes.txt'
herbarium_tag_root = 'tag/herbarium_19_tags_attributes.txt'
imagenet_tag_root = 'tag/imagenet_tags_attributes.txt'
oxford_pet_tag_root = 'tag/pets_tags_attributes_woknown.txt'
oxford_flowers_tag_root = 'tag/flowers_tags_attributes_upbond.txt'
food_101_tag_root = 'tag/food_tags_attributes_without_unkonwn.txt'