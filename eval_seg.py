import numpy as np
import argparse
import os
import pickle
import torch
from models import PointNetSeg
from data_loader import get_data_loader
from utils import create_dir, viz_seg, topk


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_seg_class', type=int, default=6, help='The number of segmentation classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='best_model')
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")

    parser.add_argument('--test_data', type=str, default='./data/seg/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/seg/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--use_saved', action="store_true")
    parser.add_argument('--exp_name', type=str, default="seg", help='The name of the experiment')
    parser.add_argument('--rotate', action="store_true")
    parser.add_argument('--rot_angle', type=float, default=0.0)
    parser.add_argument('--change_n_points', action="store_true")
    
    return parser



if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    
    if args.rotate:
        args.output_dir = f'{args.output_dir}/rot/{args.rot_angle}'
    if args.change_n_points:
        args.output_dir = f'{args.output_dir}/n_points/{args.num_points}'
    create_dir(f'{args.output_dir}/seg/best')
    create_dir(f'{args.output_dir}/seg/worst')


    # ------ TO DO: Initialize Model for Segmentation Task  ------
    model = PointNetSeg()
    
    # Load Model Checkpoint
    model_path = './checkpoints/seg/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print ("successfully loaded checkpoint from {}".format(model_path))

    # Sample Points per Object
    ind = np.random.choice(10000, args.num_points, replace=False)
    test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:])
    test_label = torch.from_numpy((np.load(args.test_label))[:,ind])

    if args.rotate:
        rot = args.rot_angle * np.pi /180
        R = torch.Tensor([
            [np.cos(rot),   -1* np.sin(rot), 0], 
            [np.sin(rot),   np.cos(rot),     0], 
            [0,              0,              1]
        ]).T
        test_data = test_data @ R

    # ------ TO DO: Make Prediction ------
    filename = f'{args.output_dir}/pred_labels_seg.pkl'
    accuracy_filename = 'pred_accuracy_seg.txt'
    use_saved = args.use_saved
    if not os.path.exists(filename) or not use_saved:
        pred_labels = []
        scores = []
        with torch.no_grad():
            for i in range(0, test_label.size()[0], 100):
                print(i)
                _, _, score = model(test_data[i:min(i + 100, test_label.size()[0]), :, :])
                score = torch.nn.functional.softmax(score, dim=-1)
                pred_label = torch.argmax(score, dim=-1)
                scores.append(score)
                pred_labels.append(pred_label)

            scores = torch.cat(scores, dim=0)
            pred_labels = torch.cat(pred_labels, dim=0)

            data = {'pred_labels': pred_labels, 'scores': scores}
            f = open(filename, 'wb')
            pickle.dump(data, f)
            f.close()
            
    f = open(filename, 'rb')
    data = pickle.load(f)
    pred_labels = data['pred_labels']
    scores = data['scores']
    f.close()

    # use the score corresponding to predicted label
    scores, _ = scores.max(dim=-1)

    # Compute Accuracy
    test_accuracy= pred_labels.eq(test_label.data).cpu().sum().item() / (test_label.reshape((-1,1)).size()[0])
    print ("test accuracy: {}".format(test_accuracy)) #test accuracy: 0.8995912479740681
    
    with open(accuracy_filename, 'a') as f:
        f.write("exp: {},  test accuracy: {} \n".format(args.output_dir, test_accuracy))
    
    test_accuracy_per_class = torch.zeros(args.num_seg_class)
    num_points_per_class = torch.zeros(args.num_seg_class)


    for idx in range(args.num_seg_class):
        cls_idx = test_label.data == idx
        test_accuracy_per_class[idx] = (pred_labels[cls_idx] == test_label.data[cls_idx]).sum()/ test_label.data[cls_idx].reshape((-1, 1)).size()[0]
        num_points_per_class[idx] = cls_idx.reshape(-1, 1).shape[0]
    # [tensor(0.7186), tensor(0.0178), tensor(0.9182), tensor(0.7526), tensor(0.8952), tensor(0.9260)]
    test_accuracy_per_object = torch.zeros(pred_labels.shape[0])
    for idx in range(len(pred_labels)):
        test_accuracy_per_object[idx] = (pred_labels[idx] == test_label.data[idx]).sum()/ test_label.data[idx].reshape((-1, 1)).size()[0]

    # find topk highest score true and false positives
    topk_best_scores, topk_best_indices = topk(test_accuracy_per_object, k=10, largest=True)
    topk_worst_scores, topk_worst_indices = topk(test_accuracy_per_object, k=10, largest=False) 

    # ------ visualize topk results ----------

    print("rendering topk best scores ..")
    viz_seg(test_data[topk_best_indices, :], pred_labels[topk_best_indices, :], test_label[topk_best_indices, :], test_accuracy_per_object[topk_best_indices], "{}/{}/best/".format(args.output_dir, args.exp_name), args.device, args.num_points)

    print("rendering topk worst scores ..")
    viz_seg(test_data[topk_worst_indices, :], pred_labels[topk_worst_indices, :], test_label[topk_worst_indices, :], test_accuracy_per_object[topk_worst_indices], "{}/{}/worst/".format(args.output_dir, args.exp_name), args.device, args.num_points)

