import json
import os
import numpy as np
import random
import pysparkling
import argparse
from .scene import Scenes
from .get_type import trajectory_type
from trajnetplusplustools import TrackRow
from .car import Car


def read_json(line):
    line = json.loads(line)
    track = line.get('track')
    if track is not None:
        return TrackRow(track['f'], track['c'], track['x'], track['y'], track['xVelocity'], track['yVelocity'],
                        track.get('prediction_number'))
    return None


def get_trackrows(sc, input_file):
    print('processing ' + input_file)
    return (sc
            .textFile(input_file)
            .map(read_json)
            .filter(lambda r: r is not None)
            .cache())


def write(input_rows, output_file, args):
    """ Write Valid Scenes without categorization """

    print(" Entering Writing ")
    # To handle two different time stamps 7:00 and 17:00 of cff
    if args.order_frames:
        frames = sorted(set(input_rows.map(lambda r: r.frame).toLocalIterator()),
                        key=lambda frame: frame % 100000)
    else:
        frames = sorted(set(input_rows.map(lambda r: r.frame).toLocalIterator()))

    # split
    train_split_index = int(len(frames) * args.train_fraction)
    val_split_index = train_split_index + int(len(frames) * args.val_fraction)
    train_frames = set(frames[:train_split_index])
    val_frames = set(frames[train_split_index:val_split_index])
    test_frames = set(frames[val_split_index:])

    # train dataset
    train_rows = input_rows.filter(lambda r: r.frame in train_frames)
    train_output = output_file.format(split='train')
    train_scenes = Scenes(fps=args.fps, start_scene_id=0, args=args).rows_to_file(train_rows, train_output)

    # validation dataset
    val_rows = input_rows.filter(lambda r: r.frame in val_frames)
    val_output = output_file.format(split='val')
    val_scenes = Scenes(fps=args.fps, start_scene_id=train_scenes.scene_id, args=args).rows_to_file(val_rows, val_output)

    # public test dataset
    test_rows = input_rows.filter(lambda r: r.frame in test_frames)
    test_output = output_file.format(split='test')
    test_scenes = Scenes(fps=args.fps, start_scene_id=val_scenes.scene_id, args=args)  # !!! Chunk Stride
    test_scenes.rows_to_file(test_rows, test_output)

    # private test dataset
    private_test_output = output_file.format(split='test_private')
    private_test_scenes = Scenes(fps=args.fps, start_scene_id=val_scenes.scene_id, args=args)
    private_test_scenes.rows_to_file(test_rows, private_test_output)


def categorize(sc, input_file, args):
    """ Categorize the Scenes """

    print(" Entering Categorizing ")
    test_fraction = 1 - args.train_fraction - args.val_fraction

    train_id = 0
    if args.train_fraction:
        print("Categorizing Training Set")
        train_rows = get_trackrows(sc, input_file.replace('split', '').format('train'))
        train_id = trajectory_type(train_rows, input_file.replace('split', '').format('train'),
                                   fps=args.fps, track_id=0, args=args)

    val_id = train_id
    if args.val_fraction:
        print("Categorizing Validation Set")
        val_rows = get_trackrows(sc, input_file.replace('split', '').format('val'))
        val_id = trajectory_type(val_rows, input_file.replace('split', '').format('val'),
                                 fps=args.fps, track_id=train_id, args=args)

    if test_fraction:
        print("Categorizing Test Set")
        test_rows = get_trackrows(sc, input_file.replace('split', '').format('test_private'))
        _ = trajectory_type(test_rows, input_file.replace('split', '').format('test_private'),
                            fps=args.fps, track_id=val_id, args=args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--obs_len', type=int, default=25,
                        help='Length of observation')
    parser.add_argument('--pred_len', type=int, default=25,
                        help='Length of prediction')
    parser.add_argument('--train_fraction', default=0.6, type=float,
                        help='Training set fraction')
    parser.add_argument('--val_fraction', default=0.2, type=float,
                        help='Validation set fraction')
    parser.add_argument('--fps', default=25, type=float,
                        help='fps')
    parser.add_argument('--order_frames', action='store_true',
                        help='For CFF')
    parser.add_argument('--chunk_stride', type=int, default=2,
                        help='Sampling Stride')
    parser.add_argument('--min_length', default=0.0, type=float,
                        help='Min Length of Primary Trajectory')
    parser.add_argument('--synthetic', action='store_true',
                        help='convert synthetic datasets (if false, convert real)')
    parser.add_argument('--direct', action='store_true',
                        help='directy convert synthetic datasets using commandline')
    parser.add_argument('--all_present', action='store_true',
                        help='filter scenes where all pedestrians present at all times')
    parser.add_argument('--orca_file', default=None,
                        help='Txt file for ORCA trajectories, required in direct mode')
    parser.add_argument('--goal_file', default=None,
                        help='Pkl file for goals (required for ORCA sensitive scene filtering)')
    parser.add_argument('--output_filename', default=None,
                        help='name of the output dataset filename in .ndjson format, required in direct mode')
    parser.add_argument('--mode', default='default', choices=('default', 'trajnet'),
                        help='mode of ORCA scene generation (required for ORCA sensitive scene filtering)')

    # For Trajectory categorizing and filtering
    categorizers = parser.add_argument_group('categorizers')
    categorizers.add_argument('--tag', type=str, default='highD',
                              help='to specify which kind of data set it is')
    categorizers.add_argument('--static_threshold', type=float, default=1.0,
                              help='Type I static threshold')
    categorizers.add_argument('--lane_threshold', type=float, default=2,
                              help='Type II distance threshold for lane change')
    categorizers.add_argument('--linear_threshold', type=float, default=0.5,
                              help='Type III linear threshold (0.3 for Synthetic)')

    # categorizers.add_argument('--inter_pos_range', type=float, default=15,
    #                           help='Type IIId angle threshold for cone (degrees)')
    # categorizers.add_argument('--grp_dist_thresh', type=float, default=0.8,
    #                           help='Type IIIc distance threshold for group')
    # categorizers.add_argument('--grp_std_thresh', type=float, default=0.2,
    #                           help='Type IIIc std deviation for group')
    categorizers.add_argument('--acceptance', nargs='+', type=float, default=[0.1, 1, 1, 1],
                              help='acceptance ratio of different trajectory (I, II, III, IV) types')

    args = parser.parse_args()
    # Set Seed
    random.seed(42)
    np.random.seed(42)

    sc = pysparkling.Context()

    # #########################
    # # Training Set
    # #########################
    # args.train_fraction = 1.0
    # args.val_fraction = 0.0
    #
    # # 获取csv文件列表
    # train_dir = 'data/ind/train'
    # files_csv = [file for file in os.listdir(train_dir) if file.endswith('_tracks.csv')]
    #
    # # 对每个csv文件执行操作
    # for file_csv in files_csv:
    #     file_path = os.path.join(train_dir, file_csv)
    #
    #     car = Car()
    #
    #     output_file = f'output_pre/{{split}}/{file_csv[:-4]}.ndjson'
    #     output_path = output_file.replace('output_pre', 'output').format(split='train')
    #
    #     write(car.read_csv(sc, file_path), output_file, args)
    #     categorize(sc, output_file, args)
    #
    #     car.append_scaler(output_path)
    #     car.delete_temp()
    #
    # #########################
    # # Validation Set
    # #########################
    # args.train_fraction = 0.0
    # args.val_fraction = 1.0
    #
    # # 获取csv文件列表
    # val_dir = 'data/ind/val'
    # files_csv = [file for file in os.listdir(val_dir) if file.endswith('_tracks.csv')]
    #
    # # 对每个csv文件执行操作
    # for file_csv in files_csv:
    #     file_path = os.path.join(val_dir, file_csv)
    #
    #     car = Car()
    #
    #     output_file = f'output_pre/{{split}}/{file_csv[:-4]}.ndjson'
    #     output_path = output_file.replace('output_pre', 'output').format(split='val')
    #
    #     write(car.read_csv(sc, file_path), output_file, args)
    #     categorize(sc, output_file, args)
    #
    #     car.append_scaler(output_path)
    #     car.delete_temp()
    #
    #########################
    # Testing Set
    #########################
    # args.train_fraction = 0.0
    # args.val_fraction = 0.0
    # args.acceptance = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    # # hard breaking # 静止static， 直行linie， 左转left+ turn， 右转right turn， 左变道left lane,右变道right lane
    # args.chunk_stride = 1
    #
    # # 获取csv文件列表
    # test_dir = 'data/ind/test'
    # files_csv = [file for file in os.listdir(test_dir) if file.endswith('_tracks.csv')]
    # # 对每个csv文件执行操作
    # for file_csv in files_csv:
    #     file_path = os.path.join(test_dir, file_csv)
    #
    #     car = Car()
    #
        # output_file = f'output_pre/{{split}}/{file_csv[:-4]}.ndjson'
        # output_path = output_file.replace('output_pre', 'output').format(split='test')
    #
    #     write(car.read_csv(sc, file_path), output_file, args)
    #     categorize(sc, output_file, args)
    #
    #     car.append_scaler(output_path)
    #     output_path = output_file.replace('output_pre', 'output').format(split='test_private')
    #     car.append_scaler(output_path)
    #     car.delete_temp()










    # Real datasets conversion
    file_dir = 'data/cars'

    files_csv = [file for file in os.listdir(file_dir) if file.endswith('_tracks.csv')]

    for file_csv in files_csv:
        if 'highD' in file_csv:
            args.tag = 'highD'
        elif 'inD' in file_csv:
            args.tag = 'inD'

        file_path = os.path.join(file_dir, file_csv)

        car = Car()

        output_file = f'output_pre/{{split}}/{file_csv[:-4]}.ndjson'


        write(car.read_csv(sc, file_path), output_file, args)
        categorize(sc, output_file, args)

        car.delete_temp()








if __name__ == '__main__':
    main()
