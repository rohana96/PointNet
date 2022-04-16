import numpy as np
import argparse
import os
import pickle
import torch
from models import PointNet
from utils import create_dir, topk, viz_cls

def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_cls_class', type=int, default=3, help='The number of classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='best_model')
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")

    parser.add_argument('--test_data', type=str, default='./data/cls/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/cls/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--use_saved', action="store_true")
    parser.add_argument('--exp_name', type=str, default="cls", help='The name of the experiment')
    parser.add_argument('--rotate', action="store_true")
    parser.add_argument('--rot_angle', type=float, default=0.0)
    parser.add_argument('--change_n_points', action="store_true")

    return parser

# test accuracy: 0.974816369359916
if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    
    if args.rotate:
        args.output_dir = f'{args.output_dir}/rot/{args.rot_angle}'
    if args.change_n_points:
        args.output_dir = f'{args.output_dir}/n_points/{args.num_points}'
    create_dir(f'{args.output_dir}/cls/topk')
    create_dir(f'{args.output_dir}/cls/classwise')

    # ------ TO DO: Initialize Model for Classification Task ------
    model = PointNet()
    
    # Load Model Checkpoint
    model_path = './checkpoints/cls/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print ("successfully loaded checkpoint from {}".format(model_path))

    # Sample Points per Object
    ind = np.random.choice(10000, args.num_points, replace=False)
    test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:])
    test_label = torch.from_numpy(np.load(args.test_label))

    if args.rotate:
        rot = args.rot_angle * np.pi /180
        R = torch.Tensor([
            [np.cos(rot),   -1* np.sin(rot), 0], 
            [np.sin(rot),   np.cos(rot),     0], 
            [0,              0,              1]
        ]).T
        test_data = test_data @ R

    # ------ TO DO: Make Prediction ------
    filename = f'{args.output_dir}/pred_labels_cls.pkl'
    accuracy_filename = 'pred_accuracy_cls.txt'
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
            scores = scores.squeeze()
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
    scores = scores.squeeze()

    # Compute Accuracy
    test_accuracy = pred_labels.eq(test_label.data).cpu().sum().item() / (test_label.size()[0])
    print ("test accuracy: {}".format(test_accuracy))

    with open(accuracy_filename, 'a') as f:
        f.write("exp: {},  test accuracy: {} \n".format(args.output_dir, test_accuracy))

    # Find misclassified and correctly classified indices
    fp_indices = torch.nonzero(pred_labels != test_label.data).squeeze()
    tp_indices = torch.nonzero(pred_labels == test_label.data).squeeze()

    # test data corresponding to correctly classified and misclassfied indices
    tp_test_data = test_data[tp_indices]
    fp_test_data = test_data[fp_indices]

    # test labels corresponding to correctly classified and misclassfied indices
    tp_test_label = test_label[tp_indices]
    fp_test_label = test_label[fp_indices]
    
    indices_dict = {
        'fp': fp_indices,
        'tp': tp_indices
    }

    label_dict = {
        'fp': fp_test_label,
        'tp': tp_test_label
    }

    data_dict = {
        'fp': fp_test_data,
        'tp': tp_test_data
    }
    # find topk highest score true and false positives
    topk_tp_scores, topk_tp_indices = topk(scores[tp_indices], k=10, largest=True)
    topk_fp_scores, topk_fp_indices = topk(scores[fp_indices], k=10, largest=True) 

    # ------ visualize topk results ----------

    # visualize topk true positives
    print("rendering topk true positives ..")
    viz_cls(tp_test_data[topk_tp_indices], pred_labels[topk_tp_indices], tp_test_label[topk_tp_indices], scores[tp_indices], "{}/{}/topk/tp".format(args.output_dir, args.exp_name), args.device, args.num_points)

    # visualize topk false positives
    print("rendering topk false positives ..")
    viz_cls(fp_test_data[topk_fp_indices], pred_labels[topk_fp_indices], fp_test_label[topk_fp_indices], scores[fp_indices], "{}/{}/topk/fp".format(args.output_dir, args.exp_name), args.device, args.num_points)

    # ------- visualize class-wise random tp and fp ----------
    for ctg in ['tp', 'fp']:
        for idx in range(args.num_cls_class):
            cls_idx = indices_dict[ctg][label_dict[ctg] == idx]
            cls_data = test_data[cls_idx]
            cls_preds = pred_labels[cls_idx]
            cls_label = test_label[cls_idx]
            cls_scores = scores[cls_idx]
            viz_cls(cls_data, cls_preds, cls_label, cls_scores, "{}/{}/classwise/{}".format(args.output_dir, args.exp_name, ctg), args.device, args.num_points, limit_render=True, max_render=3)


