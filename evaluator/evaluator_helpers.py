from dataclasses import dataclass

@dataclass
class Metrics:
    N : int
    average_l2 : float
    final_l2 : float
    # topk_ade : float
    # topk_fde : float
    # nll : float

    def __init__(self, N, average_l2, final_l2, topk_ade=None, topk_fde=None, nll=None):
        self.N = N
        self.average_l2 = average_l2
        self.final_l2 = final_l2
        # self.topk_ade = topk_ade
        # self.topk_fde = topk_fde
        # self.nll = nll


    def __iadd__(self, other):
        self.N += other.N 
        self.average_l2 += other.average_l2 
        self.final_l2 += other.final_l2
        # self.topk_ade += other.topk_ade
        # self.topk_fde += other.topk_fde
        # self.nll += other.nll
        return self

    def avg_vals(self):
        if self.N == 0:
            return
        self.average_l2 /= self.N
        self.final_l2 /= self.N
        # self.topk_ade /= self.N
        # self.topk_fde /= self.N
        # self.nll /= self.N
    
    # def to_list(self):
    #     return [self.N, self.average_l2, self.final_l2, self.topk_ade, self.topk_fde, self.nll]
    #
    # def avg_vals_to_list(self):
    #     self.avg_vals()
    #     return [self.N, self.average_l2, self.final_l2, self.topk_ade, self.topk_fde, self.nll]
    def to_list(self):
        return [self.N, self.average_l2, self.final_l2]

    def avg_vals_to_list(self):
        self.avg_vals()
        return [self.N, self.average_l2, self.final_l2]

@dataclass
class Categories:
    highD_scenes : Metrics
    inD_scenes : Metrics

    def __init__(self, highD_scenes, inD_scenes):
        self.highD_scenes = highD_scenes
        self.inD_scenes = inD_scenes


@dataclass
class Sub_categories:
    static_highD : Metrics
    lane_highD : Metrics
    linear_highD : Metrics
    others_highD : Metrics
    static_inD : Metrics
    linear_inD : Metrics
    others_inD : Metrics

    def __init__(self, static_highD, lane_highD, linear_highD, others_highD, static_inD, linear_inD, others_inD):
        self.static_highD = static_highD
        self.lane_highD = lane_highD
        self.linear_highD = linear_highD
        self.others_highD = others_highD

        self.static_inD = static_inD
        self.linear_inD = linear_inD
        self.others_inD = others_inD
