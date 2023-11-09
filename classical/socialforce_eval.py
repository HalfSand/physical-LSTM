"""Command line tool to create a table of evaluations metrics."""
import argparse
# from collections import defaultdict, namedtuple
import pickle
from .metrics import average_l2, final_l2

from .kalman import predict_kf
from . socialforce import  predict_sf
from trajnetplusplustools import Reader
from trajnetplusplustools import TrackRow, SceneRow



class Evaluator(object):
    def __init__(self, scenes, dest_dict=None, params=None, args=None):
        self.scenes = scenes
        self.dest = dest_dict
        self.params = params

        self.average_l2 = {'N': len(scenes)}
        self.final_l2 = {'N': len(scenes)}

        self.args = args

    def aggregate(self, name, predictor, dest_type='true'):
        print('evaluating', name)

        average = 0.0
        final = 0.0

        # pred_dict = {}
        # pred_neigh_dict = {}
        # n = 0

        for _, paths in enumerate(self.scenes):
            ## select only those trajectories which interactions ##
            # rows = trajnetplusplustools.Reader.paths_to_xy(paths)
            # neigh_paths = paths[1:]
            # interaction_index = collision_avoidance(rows)
            # neigh = list(compress(neigh_paths, interaction_index))
            # paths = [paths[0]] + neigh

            if 'kf' in name:
                prediction, neigh = predictor(paths, n_predict=self.args.pred_length, obs_length=self.args.obs_length)[0]
            if 'sf' in name:
                prediction, neigh = predictor(paths, self.dest, dest_type, self.params['sf'])[0]
            # if 'orca' in name:
            #     prediction, neigh = predictor(paths, self.dest, dest_type, self.params['orca'], args=self.args)[0]

            ## visualize predictions ##
            # pred_dict['pred'] = prediction
            # pred_neigh_dict['pred'] = neigh
            # n += 1
            # if n < 17:
            #     with show.predicted_paths(paths, pred_dict, pred_neigh_paths=pred_neigh_dict):
            #         pass
            # else:
            #     break

            ## Convert numpy array to Track Rows ##
            ## Extract 1) first_frame, 2) frame_diff 3) ped_ids for writing predictions
            observed_path = paths[0]
            frame_diff = observed_path[1].frame - observed_path[0].frame
            first_frame = observed_path[self.args.obs_length-1].frame + frame_diff
            ped_id = observed_path[0].car_id

            ## make Track Rows
            prediction = [TrackRow(first_frame + i * frame_diff, ped_id, prediction[i, 0], prediction[i, 1], 0)
                          for i in range(len(prediction))]

            average_l2_a = average_l2(paths[0], prediction)
            final_l2_a = final_l2(paths[0], prediction)

            # aggregate
            average += average_l2_a
            final += final_l2_a

        average /= len(self.scenes)
        final /= len(self.scenes)

        self.average_l2[name] = average
        self.final_l2[name] = final

        return self

    def result(self):
        return self.average_l2, self.final_l2

        average /= len(self.scenes)
        final /= len(self.scenes)

        self.average_l2[name] = average
        self.final_l2[name] = final

        return self

    def result(self):
        return self.average_l2, self.final_l2


def eval(input_file, dest_file, simulator, params, type_ids, args):
    print('dataset', input_file)

    reader = Reader(input_file, scene_type='paths')
    scenes = [s for _, s in reader.scenes()]

    dest_dict = None
    if dest_file is not None:
        dest_dict = pickle.load(open(dest_file, "rb"))

    evaluator = Evaluator(scenes, dest_dict, params, args)

    ## Evaluate all
    if simulator == 'all':
        for dest_type in ['interp']:
            # evaluator.aggregate('orca' + dest_type, orca.predict, dest_type)
            evaluator.aggregate('sf' + dest_type, predict_sf, dest_type)
            evaluator.aggregate('kf', predict_kf)

    # Social Force only
    elif simulator == 'sf':
        for dest_type in ['interp']:
            evaluator.aggregate('sf' + dest_type, predict_sf, dest_type)

    # Kalman only
    elif simulator == 'kf':
        evaluator.aggregate('kf', predict_kf)

    return evaluator.result()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--obs_length', default=25, type=int,
                        help='observation length')
    parser.add_argument('--pred_length', default=25, type=int,
                        help='prediction length')

    parser.add_argument('--simulator', default='all',
                        choices=('all', 'sf', 'kalman'))

    parser.add_argument('--tau', default=0.5, type=float,
                        help='Tau of Social Force')
    parser.add_argument('--vo', default=2.1, type=float,
                        help='V0 of Social Force')
    parser.add_argument('--sigma', default=0.3, type=float,
                        help='sigma of Social Force')

    parser.add_argument('--min_dist', default=4, type=float,
                        help='MinNeighDist of ORCA')
    parser.add_argument('--react_time', default=4, type=float,
                        help='NeighReactTime of ORCA')
    parser.add_argument('--radius', default=0.6, type=float,
                        help='agent radius of ORCA')

    args = parser.parse_args()

    params = {}
    if args.simulator == 'sf' or 'all':
        params['sf'] = [args.tau, args.vo, args.sigma]
    # if args.simulator == 'orca' or 'all':
    #     params['orca'] = [args.min_dist, args.react_time, args.radius]

    print(params)
    datasets = [
        'DATA_BLOCK/ind/test/04_tracks.ndjson'
    ]
    print(datasets)

    dest_dicts = None

    filtered_ids = {}


    results = {}
    for i, dataset in enumerate(datasets):
        type_ids = None
        dataset_name = dataset.replace('DATA_BLOCK/ind/test/', '').replace('.ndjson', '')
        if dataset_name in filtered_ids:
            type_ids = filtered_ids[dataset_name]
        results[dataset_name] = eval(dataset, dest_dicts, args.simulator, params, type_ids, args)

    if args.simulator == 'all':
        print('## Average L2 [m]')
        print('{dataset:>30s} | N | SF | KF '.format(dataset=''))
        print(results)
        for dataset, (r, _) in results.items():
            print(
                '{dataset:>30s}'
                ' | {r[N]:>4}'
                ' | {r[orcainterp]:.2f}'
                ' | {r[sfinterp]:.2f}'
                ' | {r[kf]:.2f}'.format(dataset=dataset, r=r)
            )
        print('')

        print('## Final L2 [m]')
        print('{dataset:>30s} | N | SF | KF '.format(dataset=''))
        for dataset, (_, r) in results.items():
            print(
                '{dataset:>30s}'
                ' | {r[N]:>4}'
                ' | {r[orcainterp]:.2f}'
                ' | {r[sfinterp]:.2f}'
                ' | {r[kf]:.2f}'.format(dataset=dataset, r=r)
            )
    ## For Hyperparameter Tuning
    # print('params: {}, {}, {} \n'.format(*params['sf']))
    # with open(args.simulator + "_final.txt", "a") as myfile:
    #         myfile.write('params: {}, {}, {} \n'.format(*params['sf']))
    #         myfile.write('## Average L2 [m]\n')
    #         myfile.write('{dataset:>30s} |   N  | Int \n'.format(dataset=''))
    #         for dataset, (r, _) in results.items():
    #             myfile.write(
    #                         '{dataset:>30s}'
    #                         ' | {r[N]:>4}'
    #                         ' | {r[sfinterp]:.2f} \n'.format(dataset=dataset, r=r)
    #             )

    #         myfile.write('\n')
    #         myfile.write('## Final L2 [m] \n')
    #         myfile.write('{dataset:>30s} |   N  | Int \n'.format(dataset=''))
    #         for dataset, (_, r) in results.items():
    #             myfile.write(
    #                         '{dataset:>30s}'
    #                         ' | {r[N]:>4}'
    #                         ' | {r[sfinterp]:.2f} \n'.format(dataset=dataset, r=r)
    #             )
    #         myfile.write('\n \n \n')

if __name__ == '__main__':
    main()
