import json
from trajnetplusplustools import SceneRow, TrackRow


def trajnet_tracks(row):
    #     x = round(row.x, 2)
    #     y = round(row.y, 2)
    #     if row.xVelocity is not None:
    #         xVelocity = round(row.xVelocity, 2)
    #     else:
    #         xVelocity = None
    #     if row.yVelocity is not None:
    #         yVelocity = round(row.yVelocity, 2)
    #     else:
    #         yVelocity = None

    x = row.x
    y = row.y
    xVelocity = row.xVelocity
    yVelocity = row.yVelocity

    if row.prediction_number is None:
        return json.dumps({'track': {'f': row.frame, 'c': row.car_id, 'x': x, 'y': y,
                                     'xVelocity': xVelocity, 'yVelocity': yVelocity}})
    return json.dumps({'track': {'f': row.frame, 'c': row.car_id, 'x': x, 'y': y,
                                 'xVelocity': xVelocity,
                                 'yVelocity': yVelocity,
                                 'prediction_number': row.prediction_number,
                                 'scene_id': row.scene_id}})


def trajnet_scenes(row):
    return json.dumps({'scene': {'id': row.scene, 'c': row.car_id, 's': row.start, 'e': row.end,
                                 'fps': row.fps, 'tag': row.tag}})


def trajnet(row):
    if isinstance(row, TrackRow):
        return trajnet_tracks(row)
    if isinstance(row, SceneRow):
        return trajnet_scenes(row)

    raise Exception('unknown row type')
