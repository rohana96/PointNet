#! /bin/bash

python train.py --task cls --exp_name pointnet_cls
python train.py --task seg --exp_name pointnet_seg

python eval_cls.py --exp_name cls --rotate --rot_angle 60
python eval_cls.py --exp_name cls --rotate --rot_angle 120
python eval_cls.py --exp_name cls --rotate --rot_angle 180
python eval_cls.py --exp_name cls --rotate --rot_angle 0

python eval_seg.py --exp_name seg --rotate --rot_angle 60
python eval_seg.py --exp_name seg --rotate --rot_angle 120
python eval_seg.py --exp_name seg --rotate --rot_angle 180
python eval_seg.py --exp_name seg --rotate --rot_angle 0

python eval_cls.py --exp_name cls --change_n_points --num_points 100
python eval_cls.py --exp_name cls --change_n_points --num_points 1000
python eval_cls.py --exp_name cls --change_n_points --num_points 5000
python eval_cls.py --exp_name cls --change_n_points --num_points 10000

 python eval_seg.py --exp_name seg --change_n_points --num_points 100
python eval_seg.py --exp_name seg --change_n_points --num_points 1000
python eval_seg.py --exp_name seg --change_n_points --num_points 5000
python eval_seg.py --exp_name seg --change_n_points --num_points 10000