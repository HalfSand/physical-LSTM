import os
from collections import defaultdict, OrderedDict
import argparse

import pickle
from joblib import Parallel, delayed
import scipy

import trajnetplusplustools
from .design_table import Table
from .evaluator_helpers import Categories, Sub_categories, Metrics


class TrajnetEvaluator:
    def __init__(self, scenes_gt, scenes_id_gt, scenes_pred, indexes, sub_indexes, args):
        ##Ground Truth
        self.scenes_gt = scenes_gt
        self.scenes_id_gt = scenes_id_gt

        ##Prediction
        self.scenes_pred = scenes_pred

        ## Dictionary of type of trajectories
        self.indexes = indexes
        self.sub_indexes = sub_indexes

        ## Overall metrics ADE, FDE, Topk_ade, Topk_fde, NLL
        self.metrics = Metrics(*([len(scenes_gt)] + [0.0]*2))
        ## Metrics for the 2 types of trajectories and 4 types of interactions
        self.categories = Categories(*[Metrics(*([len(indexes[i])] + [0.0]*2)) for i in range(1,3)])
        self.sub_categories = Sub_categories(*[Metrics(*([len(sub_indexes[i])] + [0.0]*2)) for i in range(1,8)])

        num_predictions = 0
        for track in self.scenes_pred[0][0]:
            if track.prediction_number and track.prediction_number > num_predictions:
                num_predictions = track.prediction_number
        self.num_predictions = num_predictions

        self.pred_length = args.pred_length
        self.obs_length = args.obs_length
        self.disable_collision = args.disable_collision
        self.enable_col1 = True

    def aggregate(self):

        # average, final, average_topk_ade, average_topk_fde, average_nll = [0.0]*5
        average, final = [0.0]*2
        ## Aggregates ADE, FDE, Topk ADE-FDE for each category & sub_category
        score = {i:Metrics(*[0]*3) for i in range(1,3)}
        sub_score = {i:Metrics(*[0]*3) for i in range(1,8)}
        
        ## Iterate
        for i in range(len(self.scenes_gt)):
            ground_truth = self.scenes_gt[i]

            ## Get Keys and Sub_keys
            curr_type = None
            sub_types = []

            ## Main
            for key in list(score.keys()):
                if self.scenes_id_gt[i] in self.indexes[key]:
                    curr_type = key
                    break
            # ## Sub
            for sub_key in list(sub_score.keys()):
                if self.scenes_id_gt[i] in self.sub_indexes[sub_key]:
                    sub_types.append(sub_key)

            ## Extract Prediction Frames
            primary_tracks_all = [t for t in self.scenes_pred[i][0] if t.scene_id == self.scenes_id_gt[i]]

            ##### --------------------------------------------------- SINGLE -------------------------------------------- ####

            primary_tracks = [t for t in primary_tracks_all if t.prediction_number == 0]

            frame_gt = [t.frame for t in ground_truth[0]][-self.pred_length:]
            frame_pred = [t.frame for t in primary_tracks]

            ## To verify if same scene
            if frame_gt != frame_pred:
                print("Frame id Groud truth: ", frame_gt)
                print("Frame id Predictions: ", frame_pred)
                raise Exception('frame numbers are not consistent')

            average_l2 = trajnetplusplustools.metrics.average_l2(ground_truth[0], primary_tracks, n_predictions=self.pred_length)
            final_l2 = trajnetplusplustools.metrics.final_l2(ground_truth[0], primary_tracks)

            # Add +1 to corresponding category
            score[curr_type].N += 1
            for sub_type in sub_types:
                sub_score[sub_type].N += 1


            # aggregate FDE and ADE
            average += average_l2
            final += final_l2

            score[curr_type].average_l2 += average_l2
            score[curr_type].final_l2 += final_l2
            for sub_type in sub_types:
                sub_score[sub_type].average_l2 += average_l2
                sub_score[sub_type].final_l2 += final_l2

            ##### --------------------------------------------------- SINGLE -------------------------------------------- ####

            ##### --------------------------------------------------- Top 3 -------------------------------------------- ####

            # if self.num_predictions > 1:
            #     topk_ade, topk_fde = trajnetplusplustools.metrics.topk(primary_tracks_all, ground_truth[0], n_predictions=self.pred_length)
            #     average_topk_ade += topk_ade
            #     average_topk_fde += topk_fde
            #
            #     score[curr_type].topk_ade += topk_ade
            #     score[curr_type].topk_fde += topk_fde
            #     for sub_type in sub_types:
            #         sub_score[sub_type].topk_ade += topk_ade
            #         sub_score[sub_type].topk_fde += topk_fde
                    
            ##### --------------------------------------------------- Top 3 -------------------------------------------- ####

            ##### --------------------------------------------------- NLL -------------------------------------------- ####
            # if self.num_predictions > 48:
            #     nll = trajnetplusplustools.metrics.nll(primary_tracks_all, ground_truth[0], n_predictions=self.pred_length, n_samples=50)
            #     average_nll += nll
            #
            #     score[curr_type].nll += nll
            #     for sub_type in sub_types:
            #         sub_score[sub_type].nll += nll
            ##### --------------------------------------------------- NLL -------------------------------------------- ####

        # Adding value to dict
        self.metrics.average_l2 = average
        self.metrics.final_l2 = final
        # self.metrics.nll = average_nll
        # self.metrics.topk_ade = average_topk_ade
        # self.metrics.topk_fde = average_topk_fde

        # Main categories
        self.categories.highD_scenes = score[1]
        self.categories.inD_scenes = score[2]

        ## Sub categories
        self.sub_categories.static_highD = sub_score[1]
        self.sub_categories.lane_highD = sub_score[2]
        self.sub_categories.linear_highD = sub_score[3]
        self.sub_categories.others_highD = sub_score[4]

        self.sub_categories.static_inD = sub_score[5]
        self.sub_categories.linear_inD = sub_score[6]
        self.sub_categories.others_inD = sub_score[7]

    def result(self):
        return (self.metrics, self.categories, self.sub_categories)

    ## drop pedestrians that appear post observation
    def drop_post_obs(self, ground_truth, obs_length):
        obs_end_frame = ground_truth[0][obs_length].frame
        ground_truth = [track for track in ground_truth if track[0].frame < obs_end_frame]
        return ground_truth

def collision_test(list_sub, name, args):
    """ Simple Collision Test """
    submit_datasets = [args.path + name + '/' + f for f in list_sub if 'collision_test.ndjson' in f]
    if len(submit_datasets):
        # Scene Prediction
        reader_pred = trajnetplusplustools.Reader(submit_datasets[0], scene_type='paths')
        scenes_pred = [s for _, s in reader_pred.scenes()]

        if trajnetplusplustools.metrics.collision(scenes_pred[0][0], scenes_pred[0][1], n_predictions=args.pred_length):
            return "Fail"
        return "Pass"

    return "NA"

def eval(gt, input_file, args):
    # Ground Truth
    reader_gt = trajnetplusplustools.Reader(gt, scene_type='paths')
    scenes_gt = [s for _, s in reader_gt.scenes()]
    scenes_id_gt = [s_id for s_id, _ in reader_gt.scenes()]

    # Scene Predictions
    reader_pred = trajnetplusplustools.Reader(input_file, scene_type='paths')
    scenes_pred = [s for _, s in reader_pred.scenes()]

    ## sub_indexes, indexes is dictionary deciding which scenes are in which type
    indexes = defaultdict(list)
    sub_indexes = defaultdict(list)
    
    for scene in reader_gt.scenes_by_id:
        tags = reader_gt.scenes_by_id[scene].tag
        main_type, sub_types = tags[0], tags[1]
        indexes[main_type].append(scene)
        for sub_type in sub_types:
            sub_indexes[sub_type].append(scene)

    # Evaluate
    evaluator = TrajnetEvaluator(scenes_gt, scenes_id_gt, scenes_pred, indexes, sub_indexes, args)
    evaluator.aggregate()
    result = evaluator.result()
    return result

def trajnet_evaluate(args):
    """Evaluates test_pred against test_private"""
    model_names = [model.split('/')[-1].replace('.pkl', '') + '_modes' + str(args.modes) for model in args.output]
    labels = args.labels if args.labels is not None else model_names
    table = Table()

    for num, model_name in enumerate(model_names):
        print(model_name)

        model_preds = sorted([f for f in os.listdir(args.path + model_name) if not f.startswith('.')])
        print(model_preds)

        # Simple Collision Test (if present in test_private)
        # col_result = collision_test(model_preds, model_name, args)
        # table.add_collision_entry(labels[num], col_result)

        pred_datasets = [args.path + model_name + '/' + f for f in model_preds if 'collision_test.ndjson' not in f]
        true_datasets = [args.path.replace('pred', 'private') + f for f in model_preds if 'collision_test.ndjson' not in f]
        print('pred_datasets: ', pred_datasets)
        print('true_datasets: ', true_datasets)
        # Evaluate predicted datasets with True Datasets
        results = {pred_datasets[i].replace(args.path, '').replace('.ndjson', ''):
                   eval(true_datasets[i], pred_datasets[i], args) for i in range(len(true_datasets))}

        # Add results to Table
        final_result, sub_final_result = table.add_entry(labels[num], results)

    # Output Result Table
    table.print_table()
