import numpy as np
import pysparkling
import trajnetplusplustools.metrics
from trajnetplusplustools import Reader, SceneRow
from trajnetplusplustools.writers import trajnet
from trajnetplusplustools.kalman import predict as kalman_predict


def get_type(scene, args):
    '''
    Categorization of Single Scene
    :param scene: All trajectories as TrackRows, args
    :return: The type of the traj
    '''

    ## Get xy-coordinates from trackRows
    scene_xy = Reader.paths_to_xyv(scene)

    # Type 1
    def euclidean_distance(row1, row2):
        """Euclidean distance squared between two rows."""
        return np.sqrt((row1.x - row2.x) ** 2 + (row1.y - row2.y) ** 2)

    ## Type 2
    def linear_system(scene, obs_len, pred_len):
        '''
        return: True if the traj is linear according to Kalman
        '''
        kalman_prediction, _ = kalman_predict(scene, obs_len, pred_len)[0]
        return trajnetplusplustools.metrics.final_l2(scene[0], kalman_prediction)

    def lane_change(row1, row2):
        return abs(row1.y - row2.y)


    # Category Tags
    mult_tag = []
    sub_tag = []


    # which kind of dataset
    if args.tag == 'highD':
        # 如果args.tag是'highD'，则标签为1
        mult_tag.append(1)
        # Static
        if euclidean_distance(scene[0][0], scene[0][-1]) < args.static_threshold:
            sub_tag.append(1)
        # Lane Change
        elif lane_change(scene[0][0], scene[0][-1]) > args.lane_threshold:
            sub_tag.append(2)
        # Linear
        elif linear_system(scene, args.obs_len, args.pred_len) < args.linear_threshold:
            sub_tag.append(3)
        # Non-Linear (No explainable reason)
        else:
            sub_tag.append(4)

    elif args.tag == 'inD':
        # 如果args.tag是'inD'，则标签为2
        mult_tag.append(2)
        # Static
        if euclidean_distance(scene[0][0], scene[0][-1]) < args.static_threshold:
            sub_tag.append(5)
        # Linear
        elif linear_system(scene, args.obs_len, args.pred_len) < args.linear_threshold:
            sub_tag.append(6)
        # # Lane Change
        # elif lane_change(scene[0][0], scene[0][-1]) > args.lane_threshold:
        #     sub_tag.append(3)
        # Non-Linear (No explainable reason)
        else:
            sub_tag.append(7)

    return mult_tag[0], mult_tag, sub_tag


def write_sf(rows, path, new_scenes, new_frames):
    """ Writing scenes with categories """
    output_path = path.replace('output_pre', 'output')
    pysp_tracks = rows.filter(lambda r: r.frame in new_frames).map(trajnet)
    pysp_scenes = pysparkling.Context().parallelize(new_scenes).map(trajnet)
    pysp_scenes.union(pysp_tracks).saveAsTextFile(output_path)


def trajectory_type(rows, path, fps, track_id=0, args=None):
    """ Categorization of all scenes """

    # Read
    reader = Reader(path, scene_type='paths')
    scenes = [s for _, s in reader.scenes()]
    # Filtered Frames and Scenes
    new_frames = set()
    new_scenes = []

    start_frames = set()
    ###########################################################################
    # scenes_test helps to handle both test and test_private simultaneously
    # scenes_test correspond to Test
    ###########################################################################
    test = 'test' in path
    if test:
        path_test = path.replace('test_private', 'test')
        reader_test = Reader(path_test, scene_type='paths')
        scenes_test = [s for _, s in reader_test.scenes()]
        # Filtered Test Frames and Test Scenes
        new_frames_test = set()
        new_scenes_test = []

    # Initialize Tag Stats to be collected
    tags = {1: [], 2: []}
    mult_tags = {1: [], 2: []}
    sub_tags = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: []}
    col_count = 0

    if not scenes:
        raise Exception('No scenes found')

    for index, scene in enumerate(scenes):
        if (index + 1) % 50 == 0:
            print(index)

        # Primary Path
        car_interest = scene[0]

        # Assert Test Scene length
        if test:
            assert len(scenes_test[index][0]) >= args.obs_len, \
                'Scene Test not adequate length'

        # Get Tag
        tag, mult_tag, sub_tag = get_type(scene, args)

        if np.random.uniform() < args.acceptance[tag - 1]:
            # Check Validity

            # Update Tags
            tags[tag].append(track_id)
            for tt in mult_tag:
                mult_tags[tt].append(track_id)
            for st in sub_tag:
                sub_tags[st].append(track_id)

            # Define Scene_Tag
            scene_tag = []
            scene_tag.append(tag)
            scene_tag.append(sub_tag)

            new_frames |= set(car_interest[i].frame for i in range(len(car_interest)))
            new_scenes.append(
                SceneRow(track_id, car_interest[0].car_id,
                         car_interest[0].frame, car_interest[-1].frame,
                         fps, scene_tag))

            # Append to list of scenes_test as well if Test Set
            if test:
                new_frames_test |= set(car_interest[i].frame for i in range(args.obs_len))
                new_scenes_test.append(
                    SceneRow(track_id, car_interest[0].car_id,
                             car_interest[0].frame, car_interest[-1].frame,
                             fps, 0))

            track_id += 1

    # Writes the Final Scenes and Frames
    write_sf(rows, path, new_scenes, new_frames)
    if test:
        write_sf(rows, path_test, new_scenes_test, new_frames_test)

    # Stats

    # Number of collisions found
    print("Col Count: ", col_count)

    if scenes:
        print("Total Scenes: ", index)

        # Types:
        print("Main Tags")
        print("Type 1 highD: ", len(tags[1]), "Type 2 inD: ", len(tags[2]))
        print("Sub Tags")
        print("Statics_highD: ", len(sub_tags[1]), "Lane_highD: ", len(sub_tags[2]),
              "Linear_highD: ", len(sub_tags[3]), "Others_highD: ", len(sub_tags[4]),
              "Statics_inD: ", len(sub_tags[5]), "Linear_inD: ", len(sub_tags[6]),
              "Others_inD: ", len(sub_tags[7]))

    return track_id
