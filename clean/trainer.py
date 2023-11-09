import argparse
import logging
import socket
import sys
import time
import random
import os
import torch
import numpy as np
from prepare_data import prepare_data
from reader import Reader
from utils import center_scene, random_rotation, add_noise, drop_distant
from loss import PredictionLoss, L2Loss
from lstm import MyModel, LSTMPredictor

from pythonjsonlogger import jsonlogger


# from gridbased_pooling import GridBasedPooling
# from non_gridbased_pooling import NearestNeighborMLP, HiddenStateMLPPooling, AttentionMLPPooling
# from non_gridbased_pooling import NearestNeighborLSTM, TrajectronPooling


class Trainer(object):
    def __init__(self, model=None, criterion=None, optimizer=None, lr_scheduler=None,
                 device=None, batch_size=10, obs_length=25, pred_length=25, augment=True,
                 normalize_scene=False, save_every=5, start_length=0, obs_dropout=False,
                 augment_noise=False, val_flag=True):
        self.model = model if model is not None else MyModel()
        self.criterion = criterion if criterion is not None else PredictionLoss()
        self.optimizer = optimizer if optimizer is not None else \
            torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.lr_scheduler = lr_scheduler if lr_scheduler is not None else \
            torch.optim.lr_scheduler.StepLR(self.optimizer, 15)

        self.device = device if device is not None else torch.device('cpu')
        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)
        self.log = logging.getLogger(self.__class__.__name__)
        self.save_every = save_every

        self.batch_size = batch_size
        self.obs_length = obs_length
        self.pred_length = pred_length
        self.seq_length = self.obs_length + self.pred_length

        self.augment = augment
        self.augment_noise = augment_noise
        self.normalize_scene = normalize_scene

        self.start_length = start_length
        self.obs_dropout = obs_dropout

        self.val_flag = val_flag

        for epoch in range(start_epoch, epochs):
            if epoch % self.save_every == 0:
                state = {'epoch': epoch, 'state_dict': self.model.state_dict(),
                         'optimizer': self.optimizer.state_dict(),
                         'scheduler': self.lr_scheduler.state_dict()}
                LSTMPredictor(self.model).save(state, out + '.epoch{}'.format(epoch))
            if self.val_flag:

        state = {'epoch': epoch + 1, 'state_dict': self.model.state_dict(),
                 'optimizer': self.optimizer.state_dict(),
                 'scheduler': self.lr_scheduler.state_dict()}
        LSTMPredictor(self.model).save(state, out + '.epoch{}'.format(epoch + 1))
        LSTMPredictor(self.model).save(state, out)

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

        start_time = time.time()

        print('epoch', epoch)
        random.shuffle(scenes)
        epoch_loss = 0.0
        self.model.train()
        self.optimizer.zero_grad()

        # Initialize batch of scenes
        batch_scene = []
        batch_split = [0]

        for scene_i, (filename, scene_id, paths) in enumerate(scenes):
            scene_start = time.time()

            # make new scene
            scene = Reader.paths_to_xyv(paths)

            else:

            # Drop Distant
            scene, mask = drop_distant(scene)

            # process scene
            if self.normalize_scene:
            if self.augment:
            if self.augment_noise:
                scene = add_noise(scene, thresh=0.02, ped='neigh')

            # Augment scene to batch of scenes
            batch_scene.append(scene)
            batch_split.append(int(scene.shape[1]))

            if ((scene_i + 1) % self.batch_size == 0) or ((scene_i + 1) == len(scenes)):
                # Construct Batch
                batch_scene = np.concatenate(batch_scene, axis=1)
                batch_split = np.cumsum(batch_split)

                batch_scene = torch.Tensor(batch_scene).to(self.device)
                batch_split = torch.Tensor(batch_split).to(self.device).long()

                preprocess_time = time.time() - scene_start

                # Train Batch
                loss = self.train_batch(batch_scene, batch_split)
                epoch_loss += loss
                total_time = time.time() - scene_start

                # Reset Batch
                batch_scene = []
                batch_split = [0]

            if (scene_i + 1) % (10 * self.batch_size) == 0:
                self.log.info({
                    'type': 'train',
                    'epoch': epoch, 'batch': scene_i, 'n_batches': len(scenes),
                    'time': round(total_time, 3),
                    'data_time': round(preprocess_time, 3),
                    'lr': self.get_lr(),
                    'loss': round(loss, 3),
                })

        self.lr_scheduler.step()
        self.log.info({
            'type': 'train-epoch',
            'epoch': epoch + 1,
            'loss': round(epoch_loss / (len(scenes)), 5),
            'time': round(time.time() - start_time, 1),
        })

        eval_start = time.time()

        val_loss = 0.0
        test_loss = 0.0
        self.model.train()

        # Initialize batch of scenes
        batch_scene = []
        batch_split = [0]

        for scene_i, (filename, scene_id, paths) in enumerate(scenes):
            # make new scene
            scene = Reader.paths_to_xyv(paths)

            else:

            # Drop Distant
            scene, mask = drop_distant(scene)

            # process scene
            if self.normalize_scene:

            # Augment scene to batch of scenes
            batch_scene.append(scene)
            batch_split.append(int(scene.shape[1]))

            if ((scene_i + 1) % self.batch_size == 0) or ((scene_i + 1) == len(scenes)):
                # Construct Batch
                batch_scene = np.concatenate(batch_scene, axis=1)
                batch_split = np.cumsum(batch_split)

                batch_scene = torch.Tensor(batch_scene).to(self.device)
                batch_split = torch.Tensor(batch_split).to(self.device).long()

                loss_val_batch = self.val_batch(batch_scene, batch_split)
                val_loss += loss_val_batch
                # test_loss += loss_test_batch

                # Reset Batch
                batch_scene = []
                batch_split = [0]

        eval_time = time.time() - eval_start

        self.log.info({
            'type': 'val-epoch',
            'epoch': epoch + 1,
            'loss': round(val_loss / (len(scenes)), 3),
            # 'test_loss': round(test_loss / len(scenes), 3),
            'time': round(eval_time, 1),
        })

    def train_batch(self, batch_scene, batch_split):
        """Training of B batches in parallel, B : batch_size

        Parameters
        ----------
        batch_scene : Tensor [seq_length, num_tracks_cars, 4]
            Tensor of batch of scenes.
        batch_split : Tensor [batch_size + 1]
            Tensor defining the split of the batch.
            Required to identify the tracks of to the same scene

        Returns
        -------
        loss : scalar
            Training loss of the batch
        """

        # If observation dropout active
        if self.obs_dropout:
            self.start_length = random.randint(0, self.obs_length - 2)

        observed = batch_scene[self.start_length:self.obs_length].clone()
        prediction_truth = batch_scene[self.obs_length:self.seq_length - 1].clone()

        train_pred, train_truth = self.model(observed, prediction_truth, batch_split)

        # Loss wrt primary tracks of each scene only
        loss = self.criterion(train_pred, train_truth) * self.batch_size

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def val_batch(self, batch_scene, batch_split):
        """Validation of B batches in parallel, B : batch_size

        Parameters
        ----------
        batch_scene : Tensor [seq_length, num_tracks, 2]
            Tensor of batch of scenes.
        batch_split : Tensor [batch_size + 1]
            Tensor defining the split of the batch.
            Required to identify the tracks of to the same scene

        Returns
        -------
        loss : scalar
            Validation loss of the batch when groundtruth of neighbours
            is provided
        loss_test : scalar
            Validation loss of the batch when groundtruth of neighbours
            is not provided (evaluation scenario)
        """

        if self.obs_dropout:
            self.start_length = 0

        observed = batch_scene[self.start_length:self.obs_length]
        prediction_truth = batch_scene[self.obs_length:self.seq_length - 1].clone()  # CLONE
        targets = batch_scene[self.obs_length:self.seq_length] - batch_scene[self.obs_length - 1:self.seq_length - 1]
        observed_test = observed.clone()

        with torch.no_grad():
            # groundtruth of neighbours provided (Better validation curve to monitor model)
            val_pred, val_truth = self.model(observed, prediction_truth, batch_split)
            loss = self.criterion(val_pred, val_truth) * self.batch_size


        return loss.item()


def main(epochs=25):
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=epochs, type=int,
                        help='number of epochs')
    parser.add_argument('--save_every', default=5, type=int,
                        help='frequency of saving model (in terms of epochs)')
    parser.add_argument('--obs_length', default=25, type=int,
                        help='observation length')
    parser.add_argument('--pred_length', default=25, type=int,
                        help='prediction length')
    parser.add_argument('--start_length', default=0, type=int,
                        help='starting time step of encoding observation')
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='initial learning rate')
    parser.add_argument('--step_size', default=10, type=int,
                        help='step_size of lr scheduler')
    parser.add_argument('-o', '--output', default=None,
                        help='output file')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='disable CUDA')
    parser.add_argument('--path', default='trajdata',
                        help='glob expression for data files')
    parser.add_argument('--loss', default='pred', choices=('L2', 'pred'),
                        help='loss objective, L2 loss (L2) and Gaussian loss (pred)')
    parser.add_argument('--type', default='vanilla',
                        choices=('vanilla', 'occupancy', 'directional', 'social', 'hiddenstatemlp',
                                 'nn', 'attentionmlp', 'nn_lstm', 'traj_pool'),
                        help='type of interaction encoder')
    parser.add_argument('--sample', default=1.0, type=float,
                        help='sample ratio when loading train/val scenes')
    parser.add_argument('--seed', type=int, default=42)

    # Augmentations
    parser.add_argument('--augment', action='store_true',
                        help='perform rotation augmentation')
    parser.add_argument('--normalize_scene', action='store_true',
                        help='rotate scene so primary pedestrian moves northwards at end of observation')
    parser.add_argument('--augment_noise', action='store_true',
                        help='flag to add noise to observations for robustness')
    parser.add_argument('--obs_dropout', action='store_true',
                        help='perform observation length dropout')

    # Loading pre-trained models
    pretrain = parser.add_argument_group('pretraining')
    pretrain.add_argument('--load-state', default=None,
                          help='load a pickled model state dictionary before training')
    pretrain.add_argument('--load-full-state', default=None,
                          help='load a pickled full state dictionary before training')
    pretrain.add_argument('--nonstrict-load-state', default=None,
                          help='load a pickled state dictionary before training')

    # Sequence Encoder Hyperparameters
    hyperparameters = parser.add_argument_group('hyperparameters')
    hyperparameters.add_argument('--hidden-dim', type=int, default=128,
                                 help='LSTM hidden dimension')
    hyperparameters.add_argument('--coordinate-embedding-dim', type=int, default=64,
                                 help='coordinate embedding dimension')
    hyperparameters.add_argument('--pool_dim', type=int, default=256,
                                 help='output dimension of interaction vector')

    # Grid-based pooling
    hyperparameters.add_argument('--cell_side', type=float, default=0.6,
                                 help='cell size of real world (in m) for grid-based pooling')
    hyperparameters.add_argument('--n', type=int, default=12,
                                 help='number of cells per side for grid-based pooling')
    hyperparameters.add_argument('--layer_dims', type=int, nargs='*', default=[512],
                                 help='interaction module layer dims for gridbased pooling')
    hyperparameters.add_argument('--embedding_arch', default='one_layer',
                                 help='interaction encoding arch for gridbased pooling')
    hyperparameters.add_argument('--pool_constant', default=0, type=int,
                                 help='background value (when cell empty) of gridbased pooling')
    hyperparameters.add_argument('--norm_pool', action='store_true',
                                 help='normalize the scene along direction of movement during grid-based pooling')
    hyperparameters.add_argument('--front', action='store_true',
                                 help='flag to only consider pedestrian in front during grid-based pooling')
    hyperparameters.add_argument('--latent_dim', type=int, default=16,
                                 help='latent dimension of encoding hidden dimension during social pooling')
    hyperparameters.add_argument('--norm', default=0, type=int,
                                 help='normalization scheme for input batch during grid-based pooling')

    # Non-Grid-based pooling
    hyperparameters.add_argument('--no_vel', action='store_true',
                                 help='flag to not consider relative velocity of neighbours')
    hyperparameters.add_argument('--spatial_dim', type=int, default=32,
                                 help='embedding dimension for relative position')
    hyperparameters.add_argument('--vel_dim', type=int, default=32,
                                 help='embedding dimension for relative velocity')
    hyperparameters.add_argument('--neigh', default=4, type=int,
                                 help='number of nearest neighbours to consider')
    hyperparameters.add_argument('--mp_iters', default=5, type=int,
                                 help='message passing iterations in NMMP')

    # Collision Loss
    hyperparameters.add_argument('--col_wt', default=0., type=float,
                                 help='collision loss weight')
    hyperparameters.add_argument('--col_distance', default=0.2, type=float,
                                 help='distance threshold post which collision occurs')
    args = parser.parse_args()

    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Define location to save trained model
    if not os.path.exists('OUTPUT_BLOCK/{}'.format(args.path)):
        os.makedirs('OUTPUT_BLOCK/{}'.format(args.path))
    else:
        args.output = 'OUTPUT_BLOCK/{}/lstm_{}_{}.pkl'.format(args.path, args.type, args.output)

    # configure logging
    if args.load_full_state:
        file_handler = logging.FileHandler(args.output + '.log', mode='a')
    else:
        file_handler = logging.FileHandler(args.output + '.log', mode='w')
    file_handler.setFormatter(jsonlogger.JsonFormatter('%(message)s %(levelname)s %(name)s %(asctime)s'))
    stdout_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(level=logging.INFO, handlers=[stdout_handler, file_handler])
    logging.info({
        'type': 'process',
        'argv': sys.argv,
        'args': vars(args),
        'version': version,
        'hostname': socket.gethostname(),
    })

    # refactor args for --load-state
    # loading a previously saved model
    args.load_state_strict = True
    if args.nonstrict_load_state:
        args.load_state = args.nonstrict_load_state
        args.load_state_strict = False
    if args.load_full_state:
        args.load_state = args.load_full_state

    # add args.device
    args.device = torch.device('cpu')
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')

    args.path = 'DATA_BLOCK/' + args.path
    # Prepare data

    # pretrained pool model (if any)
    pretrained_pool = None

    # # create interaction/pooling modules
    # pool = None
    # if args.type == 'hiddenstatemlp':
    #     pool = HiddenStateMLPPooling(hidden_dim=args.hidden_dim, out_dim=args.pool_dim,
    #                                  mlp_dim_vel=args.vel_dim)
    # elif args.type == 'attentionmlp':
    #     pool = AttentionMLPPooling(hidden_dim=args.hidden_dim, out_dim=args.pool_dim,
    #                                mlp_dim_spatial=args.spatial_dim, mlp_dim_vel=args.vel_dim)
    # elif args.type == 'nn':
    #     pool = NearestNeighborMLP(n=args.neigh, out_dim=args.pool_dim, no_vel=args.no_vel)
    # elif args.type == 'nn_lstm':
    #     pool = NearestNeighborLSTM(n=args.neigh, hidden_dim=args.hidden_dim, out_dim=args.pool_dim)
    # elif args.type == 'traj_pool':
    #     pool = TrajectronPooling(hidden_dim=args.hidden_dim, out_dim=args.pool_dim)
    # elif args.type != 'vanilla':
    #     pool = GridBasedPooling(type_=args.type, hidden_dim=args.hidden_dim,
    #                             cell_side=args.cell_side, n=args.n, front=args.front,
    #                             out_dim=args.pool_dim, embedding_arch=args.embedding_arch,
    #                             constant=args.pool_constant, pretrained_pool_encoder=pretrained_pool,
    #                             norm=args.norm, layer_dims=args.layer_dims, latent_dim=args.latent_dim)
    # hyper parameters
    input_size = 8  # frame, car_id, x, y, xVelocity, yVelocity, dx, dy, dvx, dvy
    hidden_size = 2  # LSTM hidden size
    num_layers = 1  # LSTM layers
    output_size = 2  # ax, ay as output

    # create forecasting model
    model = MyModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size,
                    device= args.device)

    # optimizer and schedular
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)
    lr_scheduler = None
    if args.step_size is not None:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size)
    start_epoch = 0

    # # Loss Criterion
    # criterion = L2Loss(col_wt=args.col_wt, col_distance=args.col_distance) if args.loss == 'L2' \
    #                 else PredictionLoss(col_wt=args.col_wt, col_distance=args.col_distance)
    #
    criterion = torch.nn.MSELoss()

    # train
    if args.load_state:
        # load pretrained model.
        # useful for transfer learning
        print("Loading Model Dict")
        with open(args.load_state, 'rb') as f:
            checkpoint = torch.load(f)
        pretrained_state_dict = checkpoint['state_dict']
        model.load_state_dict(pretrained_state_dict, strict=args.load_state_strict)

        if args.load_full_state:
            # load optimizers from last training
            # useful to continue model training
            print("Loading Optimizer Dict")
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint['epoch']

    # trainer
    trainer = Trainer(model, optimizer=optimizer, lr_scheduler=lr_scheduler, device=args.device,
                      criterion=criterion, batch_size=args.batch_size, obs_length=args.obs_length,
                      pred_length=args.pred_length, augment=args.augment, normalize_scene=args.normalize_scene,
                      save_every=args.save_every, start_length=args.start_length, obs_dropout=args.obs_dropout,
                      augment_noise=args.augment_noise, val_flag=val_flag)


if __name__ == '__main__':
    main()
